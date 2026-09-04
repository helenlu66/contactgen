import torch
import torch.nn.functional as F
from kornia.geometry.linalg import inverse_transformation
from manopth import rodrigues_layer
from .opt_utils import compute_uv, compute_uv_loss


def _clamp_unit(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def _norm_contact_field(pred_p: torch.Tensor, obj_cmap: torch.Tensor) -> torch.Tensor:
    """Weighted mean |SDF| on contact-target samples, clamped to [0, 1]."""
    denom = obj_cmap.sum(dim=-1).clamp(min=1e-6)
    per_batch = (torch.abs(pred_p) * obj_cmap).sum(dim=-1) / denom
    return _clamp_unit(per_batch.mean())


def _norm_uv_field(uv_pred: torch.Tensor, obj_uv: torch.Tensor, weight=None) -> torch.Tensor:
    """Mean weighted (1 - cos sim) / 2 in [0, 1]."""
    per_point = 1.0 - torch.cosine_similarity(uv_pred, obj_uv, dim=-1)
    per_point = _clamp_unit(per_point / 2.0)
    if weight is not None:
        w = weight.clamp(min=0.0)
        return _clamp_unit((per_point * w).sum() / w.sum().clamp(min=1e-6))
    return _clamp_unit(per_point.mean())


def _norm_pene_field(pene_vals: torch.Tensor, pene_mask: torch.Tensor, eps: float) -> torch.Tensor:
    """Mean penetration depth relative to ``eps``, clamped to [0, 1]."""
    if not bool(pene_mask.any()):
        return pene_vals.new_zeros(())
    depth = (-pene_vals[pene_mask]).clamp(min=0.0)
    return _clamp_unit((depth / max(-eps, 1e-6)).mean())


def _norm_clearance_hinge(
    mesh_d: torch.Tensor,
    thresholds: torch.Tensor,
) -> torch.Tensor:
    """Worst per-vertex clearance violation relative to threshold, in [0, 1]."""
    violation = F.relu(thresholds.unsqueeze(0) - mesh_d)
    rel = violation / thresholds.unsqueeze(0).clamp(min=1e-6)
    return _clamp_unit(rel.max())


def _norm_manifold_mse(
    nc_vec: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Per-DOF MSE normalized by typical nc span squared, clamped to [0, 1]."""
    mse = (nc_vec - target).pow(2).mean()
    return _clamp_unit(mse / scale.clamp(min=1e-6))


def _norm_limit_interior(nc_vec, nc_lo, nc_hi, eps=1e-6) -> torch.Tensor:
    """Soft interior joint-limit penalty; already in [0, 1] per DOF."""
    span = nc_hi - nc_lo + eps
    norm = (nc_vec - nc_lo) / span
    return _clamp_unit((F.relu(norm - 0.85) + F.relu(0.15 - norm)).mean())


def _weighted_term(weight: float, normalized: torch.Tensor) -> torch.Tensor:
    return float(weight) * normalized


def _param_like(template, shape, batch_size, dtype, device):
    if template is None:
        return torch.zeros(shape, dtype=dtype, device=device)
    out = template.to(device=device, dtype=dtype)
    if out.ndim == 1:
        out = out.unsqueeze(0)
    if out.shape[0] == 1 and batch_size > 1:
        out = out.expand(batch_size, -1)
    return out.clone()


def _sdf_forward(model, mano_layer, query_world, global_pose, mano_pose, mano_shape, mano_trans,
                 zero_global=False):
    """ArtiHand SDF query in hand canonical frame; returns pred_p_full (B, Q, 16)."""
    batch_size = query_world.shape[0]
    if zero_global:
        gp = torch.zeros_like(global_pose)
        tr = torch.zeros_like(mano_trans)
    else:
        gp = global_pose
        tr = mano_trans
    _, frames = mano_layer(
        torch.cat((gp, mano_pose), dim=1),
        th_betas=mano_shape,
        th_trans=tr,
    )
    inv_trans = inverse_transformation(frames.reshape(-1, 4, 4)).reshape(batch_size, -1, 4, 4)
    joints = frames[:, :, :3, 3]
    root = joints[:, 0, :]
    global_rotation = rodrigues_layer.batch_rodrigues(global_pose).reshape(batch_size, 3, 3)
    query_pnts_cano = torch.matmul(
        query_world - root.unsqueeze(dim=1) - mano_trans.unsqueeze(dim=1),
        global_rotation,
    ) + root.unsqueeze(dim=1)
    pnts = model.transform_queries(query_pnts_cano, inv_trans)
    pnts = model.add_pose_feature(pnts, root, inv_trans)
    pnts = model.add_shape_feature(queries=pnts, shape_indices=None, latent_shape_code=mano_shape)
    _, pred_p_full = model.forward(pnts)
    return pred_p_full, frames


def nc_signed_clearances(
    model,
    mano_layer,
    hand_verts,
    nc_vert_mask_t,
    nc_vpart_t,
    global_pose,
    mano_pose,
    mano_shape,
    mano_trans,
):
    """Per-nc-vertex signed SDF clearance (B, Vnc); negative = inside object."""
    nc_verts = hand_verts[:, nc_vert_mask_t]
    if nc_verts.shape[1] == 0:
        return torch.zeros(hand_verts.shape[0], 0, device=hand_verts.device, dtype=hand_verts.dtype)
    pred_p_full, _ = _sdf_forward(
        model, mano_layer, nc_verts, global_pose, mano_pose, mano_shape, mano_trans,
    )
    part_idx = nc_vpart_t.view(1, -1, 1).expand(pred_p_full.shape[0], -1, 1)
    return torch.gather(pred_p_full, dim=2, index=part_idx).squeeze(-1)


def _apply_mano_pose_grad_mask(mano_pose, dof_mask):
    if mano_pose.grad is not None:
        mano_pose.grad.mul_(dof_mask.view(1, -1))


def compute_nc_joint_limits(training_aa, nc_idx, k_std=3.0, std_floor=1e-4):
    """Global per-nc-DOF bounds: mean ± k_std * std from MANO training poses."""
    return compute_pose_joint_limits(training_aa, nc_idx, k_std=k_std, std_floor=std_floor)


def compute_pose_joint_limits(training_aa, dof_idx, k_std=3.0, std_floor=1e-4):
    """Per-DOF axis-angle bounds from pose pool: mean ± k_std * std."""
    import numpy as np

    block = np.asarray(training_aa, dtype=np.float64)[:, np.asarray(dof_idx, dtype=np.int64)]
    mu = block.mean(axis=0)
    std = np.maximum(block.std(axis=0), std_floor)
    lo = (mu - k_std * std).astype(np.float32)
    hi = (mu + k_std * std).astype(np.float32)
    return lo, hi


def build_phase1_alt_dof_indices(target_fingers: tuple[str, ...]):
    """MANO DOF indices that may move during phase1-grasp-edit-alt (target fingers + thumb)."""
    import numpy as np

    return np.where(build_phase1_alt_dof_mask(target_fingers))[0].astype(np.int64)


def build_synthesis_dof_indices(target_fingers: tuple[str, ...]):
    """Deprecated alias for :func:`build_phase1_alt_dof_indices`."""
    return build_phase1_alt_dof_indices(target_fingers)


def _project_pose_joint_limits(
    mano_pose: torch.Tensor,
    dof_idx: torch.Tensor,
    pose_lo: torch.Tensor,
    pose_hi: torch.Tensor,
) -> None:
    """Clamp selected MANO pose DOFs to pool-derived bounds."""
    with torch.no_grad():
        mano_pose[:, dof_idx] = torch.clamp(mano_pose[:, dof_idx], pose_lo, pose_hi)


def knn_manifold_nc_target(nc, training_nc, k=5, eps=1e-6):
    """k-NN inverse-distance blend of training nc blocks; differentiable in nc."""
    if nc.ndim == 1:
        nc = nc.unsqueeze(0)
    batch_size = nc.shape[0]
    n_train = training_nc.shape[0]
    k = min(int(k), n_train)
    diff = training_nc.unsqueeze(0) - nc.unsqueeze(1)
    dists = torch.norm(diff, dim=2)
    _, topk_idx = torch.topk(dists, k=k, largest=False, dim=1)
    topk_d = torch.gather(dists, 1, topk_idx)
    neighbors = training_nc[topk_idx.reshape(-1)].reshape(batch_size, k, -1)
    weights = 1.0 / (topk_d + eps)
    weights = weights / weights.sum(dim=1, keepdim=True)
    return (neighbors * weights.unsqueeze(-1)).sum(dim=1)


def knn_nc_centroid_target(query_finger, training_finger, training_nc, k=5):
    """k-NN by contact+nc finger block; return mean nc block of neighbors."""
    if query_finger.ndim == 1:
        query_finger = query_finger.unsqueeze(0)
    batch_size = query_finger.shape[0]
    n_train = training_finger.shape[0]
    k = min(int(k), n_train)
    diff = training_finger.unsqueeze(0) - query_finger.unsqueeze(1)
    dists = torch.norm(diff, dim=2)
    _, topk_idx = torch.topk(dists, k=k, largest=False, dim=1)
    neighbors_nc = training_nc[topk_idx.reshape(-1)].reshape(batch_size, k, -1)
    return neighbors_nc.mean(dim=1)


def _project_mano_pose(mano_pose, nc_idx, nc_lo, nc_hi, contact_dof_mask, phase1_pose):
    """Clamp nc DOFs to joint limits and pin contact DOFs to phase 1."""
    with torch.no_grad():
        mano_pose[:, nc_idx] = torch.clamp(mano_pose[:, nc_idx], nc_lo, nc_hi)
        if contact_dof_mask.dtype == torch.bool:
            contact_idx = torch.where(contact_dof_mask)[0]
        else:
            contact_idx = torch.where(contact_dof_mask > 0.5)[0]
        mano_pose[:, contact_idx] = phase1_pose[:, contact_idx]


def evaluate_nc_sdf_stats(model, mano_layer, obj_verts, original_partition, non_contact_part_ids,
                          global_pose, mano_pose, mano_shape, mano_trans):
    """Return mean nc SDF (m), fraction negative, and min nc SDF on nc-labeled samples."""
    nc_part_tensor = torch.as_tensor(non_contact_part_ids, dtype=torch.long, device=obj_verts.device)
    nc_sample_mask = torch.isin(original_partition, nc_part_tensor)
    with torch.no_grad():
        pred_p_full, _ = _sdf_forward(
            model, mano_layer, obj_verts, global_pose, mano_pose, mano_shape, mano_trans,
        )
        sdf = torch.gather(
            pred_p_full, dim=2, index=original_partition.unsqueeze(dim=-1),
        ).squeeze(-1)
        nc_sdf = sdf[nc_sample_mask]
    if nc_sdf.numel() == 0:
        return 0.0, 0.0, 0.0
    return (
        float(nc_sdf.mean().item()),
        float((nc_sdf < 0).float().mean().item()),
        float(nc_sdf.min().item()),
    )


def optimize_pose(model, mano_layer, obj_verts, obj_cmap, obj_partition, obj_uv,
                  w_contact=1e-1, w_pene=3.0, w_uv=1e-2, w_pose_reg=1e-2, w_shape_reg=1e-2,
                  global_iter=200, pose_iter=1000,
                  global_lr=5e-2, pose_lr=5e-3, eps=-1e-3,
                  init_global_pose=None, init_mano_trans=None,
                  init_mano_pose=None, init_mano_shape=None,
                  freeze_shape=False, obj_uv_weight=None):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    batch_size = obj_verts.shape[0]
    dtype = obj_verts.dtype
    device = obj_verts.device

    def _param_like(template, shape):
        if template is None:
            return torch.zeros(shape, dtype=dtype, device=device)
        out = template.to(device=device, dtype=dtype)
        if out.ndim == 1:
            out = out.unsqueeze(0)
        if out.shape[0] == 1 and batch_size > 1:
            out = out.expand(batch_size, -1)
        return out.clone()

    global_pose = _param_like(init_global_pose, (batch_size, 3))
    mano_trans = _param_like(init_mano_trans, (batch_size, 3))
    mano_pose = _param_like(init_mano_pose, (batch_size, mano_layer.ncomps))
    mano_shape = _param_like(init_mano_shape, (batch_size, 10))

    mano_pose.requires_grad = False
    mano_shape.requires_grad = False
    global_pose.requires_grad = True
    mano_trans.requires_grad = True 
    hand_opt_params = [global_pose, mano_trans]
    global_optimizer = torch.optim.Adam(hand_opt_params, lr=global_lr)
    
    for it in range(global_iter):
        loss_info = ""
        loss = 0

        _, frames = mano_layer(torch.cat((torch.zeros_like(global_pose, device=global_pose.device, dtype=global_pose.dtype), mano_pose), dim=1),
                               th_betas=mano_shape, th_trans=torch.zeros_like(mano_trans, device=mano_trans.device, dtype=mano_trans.dtype))
        inv_trans = inverse_transformation(frames.reshape(-1, 4, 4)).reshape(batch_size, -1, 4, 4)
        joints = frames[:, :, :3, 3]
        inv_trans_mat = inv_trans
        root = joints[:, 0, :]

        global_rotation = rodrigues_layer.batch_rodrigues(global_pose).reshape(batch_size, 3, 3)
        query_pnts_cano = torch.matmul(obj_verts - root.unsqueeze(dim=1) - mano_trans.unsqueeze(dim=1), global_rotation) + root.unsqueeze(dim=1)
        pnts = model.transform_queries(query_pnts_cano, inv_trans_mat)
        pnts = model.add_pose_feature(pnts, root, inv_trans_mat)
        pnts = model.add_shape_feature(queries=pnts, shape_indices=None, latent_shape_code=mano_shape)
        pred, pred_p_full = model.forward(pnts)
        
        pred_p = torch.gather(pred_p_full, dim=2, index=obj_partition.unsqueeze(dim=-1)).squeeze(-1)
        loss_contact = w_contact * (torch.abs(pred_p) * obj_cmap).sum(dim=-1).mean(dim=0)
        loss += loss_contact
        loss_info += "contact loss: {:.3f} | ".format(loss_contact.item())

        _, frames = mano_layer(torch.cat((global_pose, mano_pose), dim=1), th_betas=mano_shape, th_trans=mano_trans)
        uv_pred = compute_uv(frames, obj_verts, obj_partition)
        uv_w = obj_uv_weight if obj_uv_weight is not None else (1.0 + obj_cmap)
        uv_loss = w_uv * compute_uv_loss(uv_pred, obj_uv, weight=uv_w)
        loss += uv_loss
        loss_info += "uv loss: {:.3f}".format(uv_loss.item())

        global_optimizer.zero_grad()
        loss.backward()
        global_optimizer.step()
        print("global iter {} | ".format(it) + loss_info)

    mano_pose.requires_grad = True
    mano_shape.requires_grad = not freeze_shape
    global_pose.requires_grad = True
    mano_trans.requires_grad = True
    hand_opt_params = [global_pose, mano_pose, mano_trans]
    if not freeze_shape:
        hand_opt_params.append(mano_shape)
    pose_optimizer = torch.optim.Adam(hand_opt_params, lr=pose_lr)
    
    for it in range(pose_iter):
        loss_info = ""
        loss = 0
        _, frames = mano_layer(
            torch.cat((torch.zeros_like(global_pose, device=global_pose.device, dtype=global_pose.dtype), mano_pose),
                      dim=1),
            th_betas=mano_shape,
            th_trans=torch.zeros_like(mano_trans, device=mano_trans.device, dtype=mano_trans.dtype))
        inv_trans = inverse_transformation(frames.reshape(-1, 4, 4)).reshape(batch_size, -1, 4, 4)
        joints = frames[:, :, :3, 3]
        inv_trans_mat = inv_trans
        root = joints[:, 0, :]

        global_rotation = rodrigues_layer.batch_rodrigues(global_pose).reshape(batch_size, 3, 3)
        query_pnts_cano = torch.matmul(obj_verts - root.unsqueeze(dim=1) - mano_trans.unsqueeze(dim=1), global_rotation) + root.unsqueeze(dim=1)
        pnts = model.transform_queries(query_pnts_cano, inv_trans_mat)
        pnts = model.add_pose_feature(pnts, root, inv_trans_mat)
        pnts = model.add_shape_feature(queries=pnts, shape_indices=None, latent_shape_code=mano_shape)
        pred, pred_p_full = model.forward(pnts)
        pred_p = torch.gather(pred_p_full, dim=2, index=obj_partition.unsqueeze(dim=-1)).squeeze(-1)  # (B, Q)
        loss_contact = w_contact * (torch.abs(pred_p) * obj_cmap).sum(dim=-1).mean(dim=0) 
        loss += loss_contact
        loss_info += "contact loss: {:.3f} | ".format(loss_contact.item())

        mask = pred_p_full < eps
        masked_value = pred_p_full[mask]
        if len(masked_value) > 0:
            loss_pene = w_pene * (-masked_value.sum()) / batch_size
            loss += loss_pene
            loss_info += "pene loss: {:.3f} | ".format(loss_pene.item())

        _, frames = mano_layer(torch.cat((global_pose, mano_pose), dim=1), th_betas=mano_shape, th_trans=mano_trans)
        uv_pred = compute_uv(frames, obj_verts, obj_partition)
        uv_w = obj_uv_weight if obj_uv_weight is not None else (1 + obj_cmap)
        uv_loss = w_uv * compute_uv_loss(uv_pred, obj_uv, weight=uv_w)
        loss += uv_loss
        loss_info += "uv loss: {:.3f} | ".format(uv_loss.item())

        pose_reg_loss = w_pose_reg * (mano_pose ** 2).sum() / batch_size
        loss += pose_reg_loss
        loss_info += "pose reg loss: {:.3f} | ".format(pose_reg_loss.item())

        if not freeze_shape:
            shape_reg_loss = w_shape_reg * (mano_shape ** 2).sum() / batch_size
            loss += shape_reg_loss
            loss_info += "shape reg loss: {:.3f}".format(shape_reg_loss.item())
        else:
            loss_info += "shape frozen"

        pose_optimizer.zero_grad()
        loss.backward()
        pose_optimizer.step()
        print("iter {} | ".format(it) + loss_info)

    return global_pose, mano_pose, mano_shape, mano_trans


def optimize_grasp_edit_phase1(
    model,
    mano_layer,
    obj_verts,
    obj_cmap,
    obj_partition,
    obj_uv,
    contact_dof_mask,
    contact_part_ids,
    w_contact=1e-1,
    w_uv=1e-2,
    w_pene=3.0,
    eps=-1e-3,
    global_iter=200,
    pose_iter=1000,
    global_lr=5e-2,
    pose_lr=5e-3,
    init_global_pose=None,
    init_mano_trans=None,
    init_mano_pose=None,
    init_mano_shape=None,
    freeze_shape=True,
    obj_uv_weight=None,
):
    """Fit wrist + contact-finger DOFs to ablated GT contact fields; freeze non-contact DOFs."""
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    batch_size = obj_verts.shape[0]
    dtype = obj_verts.dtype
    device = obj_verts.device
    dof_mask = torch.as_tensor(contact_dof_mask, dtype=dtype, device=device).reshape(-1)
    if dof_mask.numel() != mano_layer.ncomps:
        raise ValueError(f"contact_dof_mask length {dof_mask.numel()} != ncomps {mano_layer.ncomps}")
    contact_part_tensor = torch.as_tensor(contact_part_ids, dtype=torch.long, device=device)

    global_pose = _param_like(init_global_pose, (batch_size, 3), batch_size, dtype, device)
    mano_trans = _param_like(init_mano_trans, (batch_size, 3), batch_size, dtype, device)
    mano_pose = _param_like(init_mano_pose, (batch_size, mano_layer.ncomps), batch_size, dtype, device)
    mano_shape = _param_like(init_mano_shape, (batch_size, 10), batch_size, dtype, device)

    mano_pose.requires_grad = False
    mano_shape.requires_grad = False
    global_pose.requires_grad = True
    mano_trans.requires_grad = True
    global_optimizer = torch.optim.Adam([global_pose, mano_trans], lr=global_lr)

    for it in range(global_iter):
        pred_p_full, _ = _sdf_forward(
            model, mano_layer, obj_verts, global_pose, mano_pose, mano_shape, mano_trans,
            zero_global=True,
        )
        pred_p = torch.gather(pred_p_full, dim=2, index=obj_partition.unsqueeze(dim=-1)).squeeze(-1)
        loss_contact = w_contact * (torch.abs(pred_p) * obj_cmap).sum(dim=-1).mean(dim=0)
        _, frames = mano_layer(
            torch.cat((global_pose, mano_pose), dim=1), th_betas=mano_shape, th_trans=mano_trans,
        )
        uv_pred = compute_uv(frames, obj_verts, obj_partition)
        uv_w = obj_uv_weight if obj_uv_weight is not None else (1.0 + obj_cmap)
        uv_loss = w_uv * compute_uv_loss(uv_pred, obj_uv, weight=uv_w)
        loss = loss_contact + uv_loss
        global_optimizer.zero_grad()
        loss.backward()
        global_optimizer.step()
        print(
            f"phase1 global iter {it} | contact loss: {loss_contact.item():.3f} | "
            f"uv loss: {uv_loss.item():.3f}"
        )

    mano_pose.requires_grad = True
    global_pose.requires_grad = True
    mano_trans.requires_grad = True
    pose_optimizer = torch.optim.Adam([global_pose, mano_pose, mano_trans], lr=pose_lr)

    for it in range(pose_iter):
        pred_p_full, _ = _sdf_forward(
            model, mano_layer, obj_verts, global_pose, mano_pose, mano_shape, mano_trans,
            zero_global=True,
        )
        pred_p = torch.gather(pred_p_full, dim=2, index=obj_partition.unsqueeze(dim=-1)).squeeze(-1)
        loss_contact = w_contact * (torch.abs(pred_p) * obj_cmap).sum(dim=-1).mean(dim=0)

        _, frames = mano_layer(
            torch.cat((global_pose, mano_pose), dim=1), th_betas=mano_shape, th_trans=mano_trans,
        )
        uv_pred = compute_uv(frames, obj_verts, obj_partition)
        uv_w = obj_uv_weight if obj_uv_weight is not None else (1.0 + obj_cmap)
        uv_loss = w_uv * compute_uv_loss(uv_pred, obj_uv, weight=uv_w)
        pene_vals = pred_p_full[:, :, contact_part_tensor]
        pene_mask = pene_vals < eps
        if pene_mask.any():
            loss_pene = w_pene * (-pene_vals[pene_mask].sum()) / batch_size
        else:
            loss_pene = torch.zeros((), dtype=dtype, device=device)
        loss = loss_contact + uv_loss + loss_pene

        pose_optimizer.zero_grad()
        loss.backward()
        _apply_mano_pose_grad_mask(mano_pose, dof_mask)
        pose_optimizer.step()
        print(
            f"phase1 pose iter {it} | contact loss: {loss_contact.item():.3f} | "
            f"pene loss: {loss_pene.item():.3f} | uv loss: {uv_loss.item():.3f}"
        )

    return (
        global_pose.detach(),
        mano_pose.detach(),
        mano_shape.detach(),
        mano_trans.detach(),
    )


def optimize_grasp_variant(
    model,
    mano_layer,
    obj_verts,
    obj_vn,
    original_partition,
    non_contact_part_ids,
    contact_part_ids,
    phase1_mano_pose,
    global_pose,
    mano_trans,
    mano_shape,
    init_mano_pose,
    contact_dof_mask,
    training_poses,
    finger_idx,
    nc_idx,
    nc_joint_lo,
    nc_joint_hi,
    hand_faces,
    nc_vert_mask_t,
    nc_vpart_t,
    nc_clearance_thresholds_t,
    w_nc_pen=0.5,
    w_nc_contact=0.8,
    w_manifold=0.2,
    w_limit=0.1,
    use_limit_interior_pen=True,
    w_pene=0.3,
    manifold_k=5,
    manifold_interval=5,
    manifold_eps=1e-6,
    eps=-1e-3,
    phase2_iter=500,
    pose_lr=5e-3,
    early_stop_patience=20,
    early_stop_eps=1e-4,
    HandObject=None,
    contact_threshold=0.25,
    max_nc_contact_samples=0,
):
    """Optimize nc DOFs: per-vertex clearance hinge until satisfied, then stop.

    Each loss term is normalized to [0, 1] before multiplying by its weight (also
    in [0, 1]). Clearance uses a signed-SDF hinge on the worst nc vertex violation
    against per-vertex thresholds (body vs fingertip/DIP regions).
    ``loss_nc_contact`` repels capsule contact on originally nc-labeled object
    samples. Manifold regularization pulls nc DOFs toward the k-NN centroid
    (contact+nc finger block) every ``manifold_interval`` steps. Once clearance and
    nc contact limits are met, clearance/manifold/nc-contact terms are disabled and
    optimization stops.
    """
    from pytorch3d.structures import Meshes

    from .diffcontact import calculate_contact_capsule

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    batch_size = obj_verts.shape[0]
    dtype = obj_verts.dtype
    device = obj_verts.device
    dof_mask = torch.as_tensor(contact_dof_mask, dtype=dtype, device=device).reshape(-1)
    non_contact_grad_mask = 1.0 - dof_mask
    nc_idx_t = torch.as_tensor(nc_idx, dtype=torch.long, device=device)
    finger_idx_t = torch.as_tensor(finger_idx, dtype=torch.long, device=device)
    training_finger = training_poses[:, finger_idx_t]
    training_nc = training_poses[:, nc_idx_t]
    nc_span_sq = (nc_joint_hi - nc_joint_lo).pow(2).mean()
    faces_t = torch.as_tensor(hand_faces, dtype=torch.long, device=device).unsqueeze(0)

    global_pose = global_pose.detach().clone()
    mano_trans = mano_trans.detach().clone()
    mano_shape = mano_shape.detach().clone()
    phase1_pose = _param_like(phase1_mano_pose, (batch_size, mano_layer.ncomps), batch_size, dtype, device)
    mano_pose = _param_like(init_mano_pose, (batch_size, mano_layer.ncomps), batch_size, dtype, device)
    init_nc = mano_pose[:, nc_idx_t].detach().clone()

    global_pose.requires_grad = False
    mano_trans.requires_grad = False
    mano_shape.requires_grad = False
    mano_pose.requires_grad = True

    _project_mano_pose(mano_pose, nc_idx_t, nc_joint_lo, nc_joint_hi, dof_mask, phase1_pose)

    nc_part_tensor = torch.as_tensor(non_contact_part_ids, dtype=torch.long, device=device)
    nc_sample_mask = torch.isin(original_partition, nc_part_tensor)
    contact_part_tensor = torch.as_tensor(contact_part_ids, dtype=torch.long, device=device)

    pose_optimizer = torch.optim.Adam([mano_pose], lr=pose_lr)
    last_stats = {
        "min_nc_mesh_mm": 0.0,
        "mean_nc_mesh_mm": 0.0,
        "n_nc_mesh_min": 0.0,
        "n_nc_contact": 0.0,
        "n_man": 0.0,
        "n_limit": 0.0,
        "n_pene": 0.0,
        "loss_nc_mesh_min": 0.0,
        "loss_nc_contact": 0.0,
        "loss_man": 0.0,
        "loss_limit": 0.0,
        "loss_pene": 0.0,
        "clearance_satisfied": False,
        "early_stop_iter": phase2_iter,
        "manifold_refresh_iter": None,
    }
    manifold_nc_target = None
    manifold_interval = max(1, int(manifold_interval))
    for it in range(phase2_iter):
        hand_verts, _ = mano_layer(
            torch.cat((global_pose, mano_pose), dim=1),
            th_betas=mano_shape,
            th_trans=mano_trans,
        )
        if bool(nc_vert_mask_t.any()):
            mesh_d = nc_signed_clearances(
                model, mano_layer, hand_verts, nc_vert_mask_t, nc_vpart_t,
                global_pose, mano_pose, mano_shape, mano_trans,
            )
            thresholds = nc_clearance_thresholds_t.unsqueeze(0)
            clearance_violation = F.relu(thresholds - mesh_d)
            max_violation = float(clearance_violation.max().item())
            min_nc_mesh = float(mesh_d.min().item())
            mean_nc_mesh = float(mesh_d.mean().item())
            tip_mask = nc_clearance_thresholds_t > nc_clearance_thresholds_t.min()
            if bool(tip_mask.any()):
                min_nc_tip = float(mesh_d[:, tip_mask].min().item())
            else:
                min_nc_tip = min_nc_mesh
            body_mask = ~tip_mask
            if bool(body_mask.any()):
                min_nc_body = float(mesh_d[:, body_mask].min().item())
            else:
                min_nc_body = min_nc_mesh
        else:
            mesh_d = None
            max_violation = 0.0
            min_nc_mesh = float("inf")
            mean_nc_mesh = float("inf")
            min_nc_tip = float("inf")
            min_nc_body = float("inf")

        clearance_ok = mesh_d is None or bool((mesh_d >= nc_clearance_thresholds_t.unsqueeze(0)).all().item())

        nc_hits = None
        if HandObject is not None:
            with torch.no_grad():
                _, hand_frames_es = mano_layer(
                    torch.cat((global_pose, mano_pose), dim=1),
                    th_betas=mano_shape,
                    th_trans=mano_trans,
                )
                ho = HandObject(device)
                out_es = ho.forward(hand_verts.detach(), hand_frames_es, obj_verts, obj_vn)
                contacts_es = out_es["contacts_object"][0, :, 0]
                part_ids_es = out_es["partition_object"].argmax(dim=-1)[0]
                nc_hits = int(
                    (
                        (contacts_es >= contact_threshold)
                        & torch.isin(part_ids_es, nc_part_tensor)
                    ).sum().item()
                )
        contact_ok = nc_hits is None or nc_hits <= max_nc_contact_samples

        pred_p_full, _ = _sdf_forward(
            model, mano_layer, obj_verts, global_pose, mano_pose, mano_shape, mano_trans,
        )

        hand_normals = Meshes(verts=hand_verts, faces=faces_t).verts_normals_padded()
        obj_contact, _ = calculate_contact_capsule(
            hand_verts, hand_normals, obj_verts, obj_vn,
            caps_top=0.0005, caps_bot=-0.0015, caps_rad=0.003, caps_on_hand=False,
        )
        obj_c = obj_contact.squeeze(-1)
        if nc_sample_mask.any() and not contact_ok:
            n_nc_contact = _clamp_unit(obj_c[nc_sample_mask].mean())
            loss_nc_contact = _weighted_term(w_nc_contact, n_nc_contact)
        else:
            n_nc_contact = obj_c.new_zeros(())
            loss_nc_contact = obj_c.new_zeros(())

        nc_vec = mano_pose[:, nc_idx_t]
        if clearance_ok:
            n_nc_mesh_min = mesh_d.new_zeros(()) if mesh_d is not None else obj_c.new_zeros(())
            n_man = obj_c.new_zeros(())
            n_limit = obj_c.new_zeros(())
            loss_nc_mesh_min = obj_c.new_zeros(())
            loss_man = obj_c.new_zeros(())
            loss_limit = obj_c.new_zeros(())
        else:
            if mesh_d is not None:
                n_nc_mesh_min = _norm_clearance_hinge(mesh_d, nc_clearance_thresholds_t)
                loss_nc_mesh_min = _weighted_term(w_nc_pen, n_nc_mesh_min)
            else:
                n_nc_mesh_min = obj_c.new_zeros(())
                loss_nc_mesh_min = obj_c.new_zeros(())
            if (
                w_manifold > 0
                and (manifold_nc_target is None or it % manifold_interval == 0)
            ):
                query_finger = mano_pose[:, finger_idx_t]
                manifold_nc_target = knn_nc_centroid_target(
                    query_finger, training_finger, training_nc, k=manifold_k,
                ).detach()
            if w_manifold > 0 and manifold_nc_target is not None:
                n_man = _norm_manifold_mse(nc_vec, manifold_nc_target, nc_span_sq)
                loss_man = _weighted_term(w_manifold, n_man)
            else:
                n_man = obj_c.new_zeros(())
                loss_man = obj_c.new_zeros(())
            if use_limit_interior_pen:
                n_limit = _norm_limit_interior(nc_vec, nc_joint_lo, nc_joint_hi, manifold_eps)
                loss_limit = _weighted_term(w_limit, n_limit)
            else:
                n_limit = obj_c.new_zeros(())
                loss_limit = obj_c.new_zeros(())

        pene_vals = pred_p_full[:, :, contact_part_tensor]
        pene_mask = pene_vals < eps
        n_pene = _norm_pene_field(pene_vals, pene_mask, eps)
        loss_pene = _weighted_term(w_pene, n_pene)

        loss = loss_nc_mesh_min + loss_nc_contact + loss_man + loss_limit + loss_pene

        nc_drift = float((nc_vec.detach() - init_nc).pow(2).sum().sqrt().item())
        last_stats = {
            "min_nc_mesh_mm": min_nc_mesh * 1000.0,
            "min_nc_body_mesh_mm": min_nc_body * 1000.0,
            "min_nc_tip_mesh_mm": min_nc_tip * 1000.0,
            "mean_nc_mesh_mm": mean_nc_mesh * 1000.0,
            "max_clearance_violation_mm": max_violation * 1000.0,
            "n_nc_mesh_min": float(n_nc_mesh_min.item()),
            "n_nc_contact": float(n_nc_contact.item()),
            "n_man": float(n_man.item()),
            "n_limit": float(n_limit.item()),
            "n_pene": float(n_pene.item()),
            "loss_nc_mesh_min": float(loss_nc_mesh_min.item()),
            "loss_nc_contact": float(loss_nc_contact.item()),
            "loss_man": float(loss_man.item()),
            "loss_limit": float(loss_limit.item()),
            "loss_pene": float(loss_pene.item()),
            "loss_total": float(loss.item()),
            "manifold_refresh_iter": it if (it % manifold_interval == 0) else last_stats.get("manifold_refresh_iter"),
            "clearance_satisfied": clearance_ok,
            "contact_satisfied": contact_ok,
            "nc_drift_from_init": nc_drift,
            "early_stop_iter": it,
            "nc_hits": nc_hits,
        }

        if it == 0 or it == phase2_iter - 1 or (it + 1) % 100 == 0:
            print(
                f"variant iter {it} | n_mesh: {last_stats['n_nc_mesh_min']:.3f} | "
                f"n_nc_contact: {last_stats['n_nc_contact']:.3f} | "
                f"n_man: {last_stats['n_man']:.3f} | "
                f"n_limit: {last_stats['n_limit']:.3f} | "
                f"n_pene: {last_stats['n_pene']:.3f} | "
                f"loss: {last_stats['loss_total']:.3f} | "
                f"min nc body/tip: {min_nc_body * 1000:.2f}/{min_nc_tip * 1000:.2f} mm | "
                f"nc drift: {nc_drift:.4f} | "
                f"clearance_ok={clearance_ok} contact_ok={contact_ok} nc_hits={nc_hits}"
            )

        if clearance_ok and contact_ok:
            last_stats["nc_hits_early_stop"] = nc_hits
            print(
                f"  stop at iter {it}: clearance + nc contact satisfied "
                f"(body/tip min {min_nc_body * 1000:.2f}/{min_nc_tip * 1000:.2f} mm, nc_hits={nc_hits})"
            )
            break

        if not loss.requires_grad:
            continue

        pose_optimizer.zero_grad()
        loss.backward()
        _apply_mano_pose_grad_mask(mano_pose, non_contact_grad_mask)
        pose_optimizer.step()
        _project_mano_pose(mano_pose, nc_idx_t, nc_joint_lo, nc_joint_hi, dof_mask, phase1_pose)

    return global_pose, mano_pose.detach(), mano_shape, mano_trans, last_stats


_FINGER_MANO_RANGES = {
    "index": (0, 9),
    "middle": (9, 18),
    "ring": (18, 27),
    "little": (27, 36),
    "thumb": (36, 45),
}


def _cg_finger_name(finger: str) -> str:
    return "little" if finger == "pinky" else finger


def _dip_part_id(finger: str) -> int:
    from motion_feature_knobs.grasp_type_knob.contactgen.recreate_ref_frame import FINGER_PART_RANGES

    lo, hi = FINGER_PART_RANGES[_cg_finger_name(finger)]
    return int(hi)


def build_phase1_alt_dof_mask(target_fingers: tuple[str, ...]):
    """True on MANO DOFs that may move during phase1-grasp-edit-alt (target fingers + thumb)."""
    import numpy as np

    mask = np.zeros(45, dtype=bool)
    for finger in target_fingers:
        lo, hi = _FINGER_MANO_RANGES[_cg_finger_name(finger)]
        mask[lo:hi] = True
    lo, hi = _FINGER_MANO_RANGES["thumb"]
    mask[lo:hi] = True
    return mask


def build_synthesis_dof_mask(target_fingers: tuple[str, ...]):
    """Deprecated alias for :func:`build_phase1_alt_dof_mask`."""
    return build_phase1_alt_dof_mask(target_fingers)


def build_target_contact_finger_dof_indices(target_fingers: tuple[str, ...]):
    """MANO DOF indices for target contact fingers only (excludes thumb/wrist)."""
    import numpy as np

    idx: list[int] = []
    for finger in target_fingers:
        lo, hi = _FINGER_MANO_RANGES[_cg_finger_name(finger)]
        idx.extend(range(lo, hi))
    return np.asarray(idx, dtype=np.int64)


def _project_synthesis_wrist_thumb(
    global_pose,
    mano_trans,
    mano_pose,
    g0,
    t0,
    thumb0,
    *,
    wrist_rot_limit: float,
    wrist_trans_limit: float,
    thumb_pose_limit: float,
    thumb_lo: int,
    thumb_hi: int,
):
    """Hard-limit wrist/thumb deviation from the reference initialization."""
    with torch.no_grad():
        drot = global_pose - g0
        drot_norm = drot.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        global_pose.copy_(g0 + drot / drot_norm * torch.clamp(drot_norm, max=wrist_rot_limit))

        dtrans = mano_trans - t0
        dtrans_norm = dtrans.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        mano_trans.copy_(t0 + dtrans / dtrans_norm * torch.clamp(dtrans_norm, max=wrist_trans_limit))

        dthumb = mano_pose[:, thumb_lo:thumb_hi] - thumb0
        dthumb_norm = dthumb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        mano_pose[:, thumb_lo:thumb_hi] = thumb0 + dthumb / dthumb_norm * torch.clamp(
            dthumb_norm, max=thumb_pose_limit,
        )


def target_contact_part_ids(target_fingers: tuple[str, ...]) -> list[int]:
    """MANO rigid part ids for target contact fingers (all phalanges per finger)."""
    from motion_feature_knobs.grasp_type_knob.contactgen.recreate_ref_frame import parse_finger_part_ids

    names = [_cg_finger_name(f) for f in target_fingers]
    return parse_finger_part_ids(names)


def _penetration_loss(
    pred_p_full: torch.Tensor,
    part_ids_tensor: torch.Tensor,
    eps: float,
    w_pene: float,
    dtype,
    device,
) -> torch.Tensor:
    """SDF penetration on selected hand parts; normalized to [0, 1] before weighting."""
    pene_vals = pred_p_full[:, :, part_ids_tensor]
    pene_mask = pene_vals < eps
    if pene_mask.any():
        return _weighted_term(w_pene, _norm_pene_field(pene_vals, pene_mask, eps))
    return torch.zeros((), dtype=dtype, device=device)


def _finger_part_vert_mask(hand_part_label: torch.Tensor, finger: str) -> torch.Tensor:
    from motion_feature_knobs.grasp_type_knob.contactgen.recreate_ref_frame import FINGER_PART_RANGES

    lo, hi = FINGER_PART_RANGES[_cg_finger_name(finger)]
    return (hand_part_label >= lo) & (hand_part_label <= hi)


def _inter_target_finger_clearance_loss(
    hand_verts: torch.Tensor,
    hand_part_label: torch.Tensor,
    target_fingers: tuple[str, ...],
    *,
    min_clearance: float = 0.008,
) -> torch.Tensor:
    """Penalize target-finger mesh regions closer than ``min_clearance`` (proxy for self-penetration)."""
    import torch.nn.functional as F

    verts = hand_verts[0]
    terms: list[torch.Tensor] = []
    fingers = list(target_fingers)
    for i in range(len(fingers)):
        for j in range(i + 1, len(fingers)):
            ma = _finger_part_vert_mask(hand_part_label, fingers[i])
            mb = _finger_part_vert_mask(hand_part_label, fingers[j])
            if not bool(ma.any()) or not bool(mb.any()):
                continue
            min_d = torch.cdist(verts[ma], verts[mb]).min()
            terms.append(F.relu(min_clearance - min_d) / max(min_clearance, 1e-4))
    if not terms:
        return hand_verts.new_zeros(())
    return _clamp_unit(torch.stack(terms).mean())


def _target_contact_centroid_separation_loss(
    centroids: dict[str, torch.Tensor],
    *,
    min_separation: float = 0.010,
) -> torch.Tensor:
    """Require target-finger contact centroids to stay at least ``min_separation`` apart."""
    import torch.nn.functional as F

    fingers = list(centroids.keys())
    terms: list[torch.Tensor] = []
    for i in range(len(fingers)):
        for j in range(i + 1, len(fingers)):
            dist = (centroids[fingers[i]] - centroids[fingers[j]]).norm()
            terms.append(F.relu(min_separation - dist) / max(min_separation, 1e-4))
    if not terms:
        return next(iter(centroids.values())).new_zeros(()) if centroids else torch.zeros(())
    return _clamp_unit(torch.stack(terms).mean())


def _target_finger_contact_coverage(
    contacts: torch.Tensor,
    partition_oh: torch.Tensor,
    target_fingers: tuple[str, ...],
    *,
    contact_thresh: float = 0.15,
) -> torch.Tensor:
    """Penalize weak capsule contact on target-finger object samples (in [0, 1])."""
    import torch.nn.functional as F

    cmap = contacts[0, :, 0]
    part_ids = partition_oh[0].argmax(dim=-1)
    terms: list[torch.Tensor] = []
    for finger in target_fingers:
        pid = _dip_part_id(finger)
        mask = part_ids == pid
        if mask.any():
            terms.append(F.relu(contact_thresh - cmap[mask].mean()))
        else:
            terms.append(cmap.new_tensor(1.0))
    return _clamp_unit(torch.stack(terms).mean())


def _finger_contact_centroids(
    contacts: torch.Tensor,
    partition_oh: torch.Tensor,
    obj_verts: torch.Tensor,
    target_fingers: tuple[str, ...],
    *,
    contact_thresh: float = 0.08,
) -> dict[str, torch.Tensor]:
    """Differentiable contact centroids per target finger on object samples."""
    cmap = contacts[0, :, 0]
    part_ids = partition_oh[0].argmax(dim=-1)
    centroids: dict[str, torch.Tensor] = {}
    for finger in target_fingers:
        pid = _dip_part_id(finger)
        part_mask = part_ids == pid
        active = part_mask & (cmap > contact_thresh)
        if active.any():
            w = cmap[active]
            pts = obj_verts[0, active]
            centroids[finger] = (pts * w.unsqueeze(-1)).sum(dim=0) / w.sum().clamp(min=1e-6)
        elif part_mask.any():
            centroids[finger] = obj_verts[0, part_mask].mean(dim=0)
        else:
            centroids[finger] = obj_verts[0].mean(dim=0)
    return centroids


def score_target_finger_contacts(
    hand_object,
    mano_layer,
    global_pose: torch.Tensor,
    mano_pose: torch.Tensor,
    mano_shape: torch.Tensor,
    mano_trans: torch.Tensor,
    obj_verts: torch.Tensor,
    obj_vn: torch.Tensor,
    target_fingers: tuple[str, ...],
    *,
    contact_thresh: float = 0.08,
    weight_fingers: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Score how strongly each target finger contacts the object (capsule field)."""
    with torch.no_grad():
        hand_verts, hand_frames = mano_layer(
            torch.cat((global_pose, mano_pose), dim=1),
            th_betas=mano_shape,
            th_trans=mano_trans,
        )
        ho = hand_object.forward(hand_verts, hand_frames, obj_verts, obj_vn)
        cmap = ho["contacts_object"][0, :, 0]
        part_ids = ho["partition_object"][0].argmax(dim=-1)

    per_finger: dict[str, float] = {}
    for finger in target_fingers:
        pid = _dip_part_id(finger)
        mask = part_ids == pid
        if bool(mask.any()):
            per_finger[finger] = float(cmap[mask].mean().item())
        else:
            per_finger[finger] = 0.0
    active = sum(1 for v in per_finger.values() if v >= contact_thresh)
    mean_contact = sum(per_finger.values()) / max(len(per_finger), 1)
    min_contact = min(per_finger.values()) if per_finger else 0.0
    weighted_sum = 0.0
    for finger in target_fingers:
        w = 2.0 if weight_fingers is not None and finger in weight_fingers else 1.0
        weighted_sum += w * per_finger.get(finger, 0.0)
    return {
        "per_finger": per_finger,
        "mean_contact": mean_contact,
        "min_contact": min_contact,
        "n_active": float(active),
        "score": float(active) * 1000.0 + weighted_sum + min_contact * 50.0,
    }


def _point_to_axis_distance_sq(
    point: torch.Tensor,
    axis_origin: torch.Tensor,
    axis_dir: torch.Tensor,
) -> torch.Tensor:
    """Squared perpendicular distance from ``point`` to the VF opposition axis."""
    v = point.reshape(3) - axis_origin.reshape(3)
    direction = axis_dir.reshape(3)
    direction = direction / direction.norm().clamp(min=1e-9)
    proj = (v * direction).sum()
    perp = v - proj * direction
    return perp.pow(2).sum()


def optimize_grasp_edit_phase1_alt(
    model,
    mano_layer,
    hand_object,
    obj_verts,
    obj_vn,
    obj_cmap,
    obj_partition,
    obj_uv,
    target_fingers: tuple[str, ...],
    vf_axis_origin_sample: torch.Tensor,
    vf_axis_dir_sample: torch.Tensor,
    vf_axis_ref_endpoint_sample: torch.Tensor,
    ref_centroids_sample: dict[str, torch.Tensor],
    target_pene_part_ids,
    *,
    reference_finger_names: tuple[str, ...] = (),
    w_contact: float = 1.0,
    w_uv: float = 0.15,
    w_pene: float = 1.0,
    w_wrist: float = 1.0,
    w_thumb: float = 2.0,
    w_anchor: float = 0.20,
    w_ref_anchor: float = 2.0,
    w_vf: float = 1.5,
    w_coverage: float = 1.0,
    w_self_clear: float = 0.85,
    w_centroid_sep: float = 1.0,
    vf_scale: float = 0.012,
    anchor_scale: float = 0.012,
    wrist_rot_scale: float = 0.02,
    wrist_trans_scale: float = 0.005,
    thumb_pose_scale: float = 0.10,
    min_finger_clearance: float = 0.008,
    min_centroid_separation: float = 0.014,
    wrist_rot_limit: float = 0.010,
    wrist_trans_limit: float = 0.002,
    thumb_pose_limit: float = 0.08,
    coverage_thresh: float = 0.15,
    eps: float = -1e-3,
    global_iter: int = 40,
    pose_iter: int = 400,
    global_lr: float = 2e-2,
    pose_lr: float = 3e-3,
    init_global_pose=None,
    init_mano_trans=None,
    init_mano_pose=None,
    init_mano_shape=None,
    freeze_shape: bool = True,
    obj_uv_weight=None,
    training_poses=None,
    target_finger_idx=None,
    w_knn: float = 0.25,
    manifold_k: int = 5,
    manifold_interval: int = 5,
    synthesis_dof_idx=None,
    pose_lo=None,
    pose_hi=None,
):
    """Phase1-grasp-edit-alt: fit wrist + target/thumb DOFs to synthesized contact fields.

    Alternative to :func:`optimize_grasp_edit_phase1` when GT contact map / hand-part /
    UV fields are unavailable and must be synthesized first.
    """
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    batch_size = obj_verts.shape[0]
    dtype = obj_verts.dtype
    device = obj_verts.device
    dof_mask = torch.as_tensor(
        build_phase1_alt_dof_mask(target_fingers), dtype=dtype, device=device,
    ).reshape(-1)
    contact_part_tensor = torch.as_tensor(target_pene_part_ids, dtype=torch.long, device=device)
    hand_part_label = hand_object.hand_part_label.reshape(-1).long()
    vf_axis_origin = vf_axis_origin_sample.to(device=device, dtype=dtype).reshape(3)
    vf_axis_dir = vf_axis_dir_sample.to(device=device, dtype=dtype).reshape(3)
    vf_axis_dir = vf_axis_dir / vf_axis_dir.norm().clamp(min=1e-9)
    vf_axis_ref_endpoint = vf_axis_ref_endpoint_sample.to(device=device, dtype=dtype).reshape(3)

    global_pose = _param_like(init_global_pose, (batch_size, 3), batch_size, dtype, device)
    mano_trans = _param_like(init_mano_trans, (batch_size, 3), batch_size, dtype, device)
    mano_pose = _param_like(init_mano_pose, (batch_size, mano_layer.ncomps), batch_size, dtype, device)
    mano_shape = _param_like(init_mano_shape, (batch_size, 10), batch_size, dtype, device)
    g0 = global_pose.detach().clone()
    t0 = mano_trans.detach().clone()
    thumb_lo, thumb_hi = _FINGER_MANO_RANGES["thumb"]
    thumb0 = mano_pose[:, thumb_lo:thumb_hi].detach().clone()
    limit_dof_t = None
    limit_lo_t = None
    limit_hi_t = None
    if synthesis_dof_idx is not None and pose_lo is not None and pose_hi is not None:
        limit_dof_t = torch.as_tensor(synthesis_dof_idx, dtype=torch.long, device=device)
        limit_lo_t = torch.as_tensor(pose_lo, dtype=dtype, device=device).reshape(-1)
        limit_hi_t = torch.as_tensor(pose_hi, dtype=dtype, device=device).reshape(-1)

    training_target = None
    target_idx_t = None
    target_span_sq = None
    manifold_target = None
    if training_poses is not None and target_finger_idx is not None and w_knn > 0:
        target_idx_t = torch.as_tensor(target_finger_idx, dtype=torch.long, device=device)
        training_poses = training_poses.to(device=device, dtype=dtype)
        training_target = training_poses[:, target_idx_t]
        target_span_sq = (
            (training_target.max(dim=0).values - training_target.min(dim=0).values)
            .pow(2)
            .mean()
            .clamp(min=1e-4)
        )

    def _extra_losses(pred_p_full, frames, *, it: int = 0, pose_phase: bool = False):
        nonlocal manifold_target
        hand_verts, _ = mano_layer(
            torch.cat((global_pose, mano_pose), dim=1), th_betas=mano_shape, th_trans=mano_trans,
        )
        ho = hand_object.forward(hand_verts, frames, obj_verts, obj_vn)
        centroids = _finger_contact_centroids(
            ho["contacts_object"], ho["partition_object"], obj_verts, target_fingers,
        )
        coverage_norm = _target_finger_contact_coverage(
            ho["contacts_object"], ho["partition_object"], target_fingers,
            contact_thresh=coverage_thresh,
        )
        stacked = torch.stack([centroids[f] for f in target_fingers], dim=0)
        vf_achieved = stacked.mean(dim=0)
        thumb_centroids = _finger_contact_centroids(
            ho["contacts_object"], ho["partition_object"], obj_verts, ("thumb",),
            contact_thresh=coverage_thresh,
        )
        thumb_achieved = thumb_centroids["thumb"]
        thumb_axis_norm = _clamp_unit(
            (thumb_achieved - vf_axis_origin).pow(2).sum() / max(vf_scale, 1e-4) ** 2
        )
        vf_ref_norm = _clamp_unit(
            (vf_achieved - vf_axis_ref_endpoint).pow(2).sum() / max(vf_scale, 1e-4) ** 2
        )
        vf_perp_norm = _clamp_unit(
            _point_to_axis_distance_sq(vf_achieved, vf_axis_origin, vf_axis_dir) / max(vf_scale, 1e-4) ** 2
        )
        vf_norm = (thumb_axis_norm + vf_ref_norm + vf_perp_norm) / 3.0

        anchor_terms = []
        ref_anchor_terms = []
        for finger, ref_c in ref_centroids_sample.items():
            if finger == "thumb":
                continue
            if finger not in centroids:
                continue
            term = ((centroids[finger] - ref_c.to(device=device, dtype=dtype)) / max(anchor_scale, 1e-4)).pow(2).mean()
            if finger in reference_finger_names:
                ref_anchor_terms.append(term)
            else:
                anchor_terms.append(term)
        anchor_norm = _clamp_unit(torch.stack(anchor_terms).mean()) if anchor_terms else vf_achieved.new_zeros(())
        ref_anchor_norm = (
            _clamp_unit(torch.stack(ref_anchor_terms).mean())
            if ref_anchor_terms
            else vf_achieved.new_zeros(())
        )

        wrist_norm = _clamp_unit(
            (global_pose - g0).pow(2).sum() / max(wrist_rot_scale, 1e-4) ** 2
            + (mano_trans - t0).pow(2).sum() / max(wrist_trans_scale, 1e-4) ** 2
        )
        thumb_norm = _clamp_unit(
            (mano_pose[:, thumb_lo:thumb_hi] - thumb0).pow(2).mean() / max(thumb_pose_scale, 1e-4) ** 2
        )
        self_clear_norm = _inter_target_finger_clearance_loss(
            hand_verts, hand_part_label, target_fingers, min_clearance=min_finger_clearance,
        )
        centroid_sep_norm = _target_contact_centroid_separation_loss(
            centroids, min_separation=min_centroid_separation,
        )
        out = (
            _weighted_term(w_coverage, coverage_norm)
            + _weighted_term(w_vf, vf_norm)
            + _weighted_term(w_anchor, anchor_norm)
            + _weighted_term(w_ref_anchor, ref_anchor_norm)
            + _weighted_term(w_wrist, wrist_norm)
            + _weighted_term(w_thumb, thumb_norm)
            + _weighted_term(w_self_clear, self_clear_norm)
            + _weighted_term(w_centroid_sep, centroid_sep_norm)
        )
        if pose_phase and training_target is not None and w_knn > 0:
            if manifold_target is None or it % manifold_interval == 0:
                target_vec = mano_pose[:, target_idx_t]
                manifold_target = knn_nc_centroid_target(
                    target_vec, training_target, training_target, k=manifold_k,
                ).detach()
            target_vec = mano_pose[:, target_idx_t]
            n_knn = _norm_manifold_mse(target_vec, manifold_target, target_span_sq)
            out = out + _weighted_term(w_knn, n_knn)
        return out

    mano_pose.requires_grad = False
    mano_shape.requires_grad = False
    global_pose.requires_grad = True
    mano_trans.requires_grad = True
    global_optimizer = torch.optim.Adam([global_pose, mano_trans], lr=global_lr)

    for it in range(global_iter):
        pred_p_full, frames = _sdf_forward(
            model, mano_layer, obj_verts, global_pose, mano_pose, mano_shape, mano_trans,
            zero_global=False,
        )
        pred_p = torch.gather(pred_p_full, dim=2, index=obj_partition.unsqueeze(dim=-1)).squeeze(-1)
        loss_contact = _weighted_term(w_contact, _norm_contact_field(pred_p, obj_cmap))
        _, frames = mano_layer(
            torch.cat((global_pose, mano_pose), dim=1), th_betas=mano_shape, th_trans=mano_trans,
        )
        uv_pred = compute_uv(frames, obj_verts, obj_partition)
        uv_w = obj_uv_weight if obj_uv_weight is not None else (1.0 + obj_cmap)
        loss_uv = _weighted_term(w_uv, _norm_uv_field(uv_pred, obj_uv, weight=uv_w))
        loss_pene = _penetration_loss(
            pred_p_full, contact_part_tensor, eps, w_pene, dtype, device,
        )
        loss = loss_contact + loss_uv + loss_pene + _extra_losses(pred_p_full, frames)
        global_optimizer.zero_grad()
        loss.backward()
        global_optimizer.step()
        _project_synthesis_wrist_thumb(
            global_pose, mano_trans, mano_pose, g0, t0, thumb0,
            wrist_rot_limit=wrist_rot_limit,
            wrist_trans_limit=wrist_trans_limit,
            thumb_pose_limit=thumb_pose_limit,
            thumb_lo=thumb_lo, thumb_hi=thumb_hi,
        )
        if limit_dof_t is not None:
            _project_pose_joint_limits(mano_pose, limit_dof_t, limit_lo_t, limit_hi_t)
        if it % 25 == 0 or it == global_iter - 1:
            pene_stat = _norm_pene_field(
                pred_p_full[:, :, contact_part_tensor],
                pred_p_full[:, :, contact_part_tensor] < eps,
                eps,
            ).item() if (pred_p_full[:, :, contact_part_tensor] < eps).any() else 0.0
            print(
                f"phase1-alt global iter {it} | loss {loss.item():.4f} "
                f"(pene {pene_stat:.3f})"
            )

    mano_pose.requires_grad = True
    global_pose.requires_grad = False
    mano_trans.requires_grad = False
    pose_optimizer = torch.optim.Adam([mano_pose], lr=pose_lr)
    manifold_target = None

    for it in range(pose_iter):
        pred_p_full, _ = _sdf_forward(
            model, mano_layer, obj_verts, global_pose, mano_pose, mano_shape, mano_trans,
            zero_global=False,
        )
        pred_p = torch.gather(pred_p_full, dim=2, index=obj_partition.unsqueeze(dim=-1)).squeeze(-1)
        loss_contact = _weighted_term(w_contact, _norm_contact_field(pred_p, obj_cmap))

        _, frames = mano_layer(
            torch.cat((global_pose, mano_pose), dim=1), th_betas=mano_shape, th_trans=mano_trans,
        )
        uv_pred = compute_uv(frames, obj_verts, obj_partition)
        uv_w = obj_uv_weight if obj_uv_weight is not None else (1.0 + obj_cmap)
        loss_uv = _weighted_term(w_uv, _norm_uv_field(uv_pred, obj_uv, weight=uv_w))
        loss_pene = _penetration_loss(
            pred_p_full, contact_part_tensor, eps, w_pene, dtype, device,
        )

        loss = loss_contact + loss_uv + loss_pene + _extra_losses(
            pred_p_full, frames, it=it, pose_phase=True,
        )
        pose_optimizer.zero_grad()
        loss.backward()
        _apply_mano_pose_grad_mask(mano_pose, dof_mask)
        pose_optimizer.step()
        _project_synthesis_wrist_thumb(
            global_pose, mano_trans, mano_pose, g0, t0, thumb0,
            wrist_rot_limit=wrist_rot_limit,
            wrist_trans_limit=wrist_trans_limit,
            thumb_pose_limit=thumb_pose_limit,
            thumb_lo=thumb_lo, thumb_hi=thumb_hi,
        )
        if limit_dof_t is not None:
            _project_pose_joint_limits(mano_pose, limit_dof_t, limit_lo_t, limit_hi_t)
        if it % 50 == 0 or it == pose_iter - 1:
            pene_vals = pred_p_full[:, :, contact_part_tensor]
            pene_mask = pene_vals < eps
            pene_stat = _norm_pene_field(pene_vals, pene_mask, eps).item() if pene_mask.any() else 0.0
            print(
                f"phase1-alt pose iter {it} | loss {loss.item():.4f} "
                f"(contact {_norm_contact_field(pred_p, obj_cmap).item():.3f}, "
                f"pene {pene_stat:.3f})"
            )

    return (
        global_pose.detach(),
        mano_pose.detach(),
        mano_shape.detach(),
        mano_trans.detach(),
    )


def _build_new_finger_dof_mask(new_finger_names: tuple[str, ...]) -> "np.ndarray":
    import numpy as np

    mask = np.zeros(45, dtype=bool)
    for finger in new_finger_names:
        lo, hi = _FINGER_MANO_RANGES[_cg_finger_name(finger)]
        mask[lo:hi] = True
    return mask


def _finger_tip_vert_mask(hand_part_label: torch.Tensor, finger: str) -> torch.Tensor:
    from motion_feature_knobs.grasp_type_knob.contactgen.recreate_ref_frame import FINGER_PART_RANGES

    _, hi = FINGER_PART_RANGES[_cg_finger_name(finger)]
    return hand_part_label == hi


def _target_finger_mesh_reach_loss(
    hand_verts: torch.Tensor,
    obj_verts: torch.Tensor,
    hand_part_label: torch.Tensor,
    new_finger_names: tuple[str, ...],
    *,
    min_surface_dist: float = 0.0008,
    max_surface_dist: float = 0.0030,
) -> torch.Tensor:
    """Keep new-finger tips in a contact band above the object surface (Euclidean)."""
    import torch.nn.functional as F
    from pytorch3d.ops import knn_points

    terms: list[torch.Tensor] = []
    for finger in new_finger_names:
        vert_mask = _finger_tip_vert_mask(hand_part_label, finger)
        if not bool(vert_mask.any()):
            vert_mask = _finger_part_vert_mask(hand_part_label, finger)
        if not bool(vert_mask.any()):
            continue
        tips = hand_verts[:, vert_mask]
        dists = knn_points(tips, obj_verts, K=1).dists.sqrt().squeeze(-1)
        d_min = dists.min(dim=1).values
        terms.append(
            F.relu(d_min - max_surface_dist).mean() + F.relu(min_surface_dist - d_min).mean()
        )
    if not terms:
        return hand_verts.new_zeros(())
    return torch.stack(terms).mean()


def _target_finger_tip_penetration_loss(
    model,
    mano_layer,
    hand_verts: torch.Tensor,
    hand_part_label: torch.Tensor,
    finger_names: tuple[str, ...],
    global_pose: torch.Tensor,
    mano_pose: torch.Tensor,
    mano_shape: torch.Tensor,
    mano_trans: torch.Tensor,
    *,
    clearance_floor: float = 0.0,
) -> torch.Tensor:
    """Penalize target-finger tips with signed SDF clearance below ``clearance_floor``."""
    import torch.nn.functional as F

    terms: list[torch.Tensor] = []
    for finger in finger_names:
        vert_mask = _finger_tip_vert_mask(hand_part_label, finger)
        if not bool(vert_mask.any()):
            continue
        vpart = hand_part_label[vert_mask].reshape(1, -1)
        mesh_d = nc_signed_clearances(
            model,
            mano_layer,
            hand_verts,
            vert_mask,
            vpart,
            global_pose,
            mano_pose,
            mano_shape,
            mano_trans,
        )
        terms.append(F.relu(clearance_floor - mesh_d.min(dim=1).values).mean())
    if not terms:
        return hand_verts.new_zeros(())
    return _clamp_unit(torch.stack(terms).mean())


def optimize_grasp_edit_phase1_alt_reach(
    model,
    mano_layer,
    hand_object,
    obj_verts,
    obj_vn,
    obj_cmap,
    obj_partition,
    new_finger_names: tuple[str, ...],
    ref_centroids_sample: dict[str, torch.Tensor],
    target_pene_part_ids,
    *,
    reference_finger_names: tuple[str, ...] = (),
    vf_axis_origin_sample=None,
    vf_axis_ref_endpoint_sample=None,
    target_fingers: tuple[str, ...] = (),
    init_global_pose,
    init_mano_trans,
    init_mano_pose,
    init_mano_shape,
    pose_lo=None,
    pose_hi=None,
    new_finger_dof_idx=None,
    w_reach: float = 4.0,
    w_coverage: float = 2.0,
    w_anchor: float = 0.25,
    w_ref_anchor: float = 2.5,
    w_vf: float = 2.0,
    w_pene: float = 2.5,
    w_mesh_pene: float = 3.0,
    w_wrist: float = 0.35,
    min_surface_dist: float = 0.0008,
    max_surface_dist: float = 0.0030,
    coverage_thresh: float = 0.10,
    anchor_scale: float = 0.010,
    wrist_rot_scale: float = 0.04,
    wrist_trans_scale: float = 0.008,
    wrist_rot_limit: float = 0.025,
    wrist_trans_limit: float = 0.004,
    thumb_pose_limit: float = 0.10,
    eps: float = -1e-3,
    reach_iter: int = 200,
    reach_lr: float = 3e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Phase1-grasp-edit-alt reach: pull newly synthesized fingers onto the object."""
    import numpy as np
    import torch.nn.functional as F

    if not new_finger_names:
        return init_global_pose, init_mano_pose, init_mano_shape, init_mano_trans

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    batch_size = obj_verts.shape[0]
    dtype = obj_verts.dtype
    device = obj_verts.device
    hand_part_label = hand_object.hand_part_label.reshape(-1).long()
    contact_part_tensor = torch.as_tensor(target_pene_part_ids, dtype=torch.long, device=device)

    global_pose = _param_like(init_global_pose, (batch_size, 3), batch_size, dtype, device)
    mano_trans = _param_like(init_mano_trans, (batch_size, 3), batch_size, dtype, device)
    mano_pose = _param_like(init_mano_pose, (batch_size, mano_layer.ncomps), batch_size, dtype, device)
    mano_shape = _param_like(init_mano_shape, (batch_size, 10), batch_size, dtype, device)

    g0 = global_pose.detach().clone()
    t0 = mano_trans.detach().clone()
    thumb_lo, thumb_hi = _FINGER_MANO_RANGES["thumb"]
    thumb0 = mano_pose[:, thumb_lo:thumb_hi].detach().clone()
    new_dof_mask = torch.as_tensor(
        _build_new_finger_dof_mask(new_finger_names), dtype=dtype, device=device,
    )
    ref_dof_idx = (
        build_target_contact_finger_dof_indices(reference_finger_names)
        if reference_finger_names
        else np.asarray([], dtype=np.int64)
    )
    ref_dof_t = (
        torch.as_tensor(ref_dof_idx, dtype=torch.long, device=device)
        if ref_dof_idx.size
        else None
    )
    ref_pose_pin = (
        mano_pose[:, ref_dof_t].detach().clone() if ref_dof_t is not None else None
    )
    limit_dof_t = None
    limit_lo_t = None
    limit_hi_t = None
    if new_finger_dof_idx is not None and pose_lo is not None and pose_hi is not None:
        limit_dof_t = torch.as_tensor(new_finger_dof_idx, dtype=torch.long, device=device)
        limit_lo_t = torch.as_tensor(pose_lo, dtype=dtype, device=device).reshape(-1)
        limit_hi_t = torch.as_tensor(pose_hi, dtype=dtype, device=device).reshape(-1)

    global_pose.requires_grad = True
    mano_trans.requires_grad = True
    mano_pose.requires_grad = True
    mano_shape.requires_grad = False

    optimizer = torch.optim.Adam(
        [
            {"params": [global_pose, mano_trans], "lr": reach_lr * 1.5},
            {"params": [mano_pose], "lr": reach_lr},
        ]
    )

    vf_axis_origin = None
    vf_axis_ref_endpoint = None
    if vf_axis_origin_sample is not None and vf_axis_ref_endpoint_sample is not None:
        vf_axis_origin = vf_axis_origin_sample.to(device=device, dtype=dtype).reshape(3)
        vf_axis_ref_endpoint = vf_axis_ref_endpoint_sample.to(device=device, dtype=dtype).reshape(3)
    vf_scale = anchor_scale

    for it in range(reach_iter):
        hand_verts, frames = mano_layer(
            torch.cat((global_pose, mano_pose), dim=1), th_betas=mano_shape, th_trans=mano_trans,
        )
        ho = hand_object.forward(hand_verts, frames, obj_verts, obj_vn)
        coverage_terms: list[torch.Tensor] = []
        anchor_terms: list[torch.Tensor] = []
        ref_anchor_terms: list[torch.Tensor] = []
        cmap = ho["contacts_object"][0, :, 0]
        part_ids = ho["partition_object"][0].argmax(dim=-1)

        def _finger_centroid_from_ho(finger: str) -> torch.Tensor:
            pid = _dip_part_id(finger)
            mask = part_ids == pid
            active = mask & (cmap > coverage_thresh * 0.5)
            if active.any():
                w = cmap[active]
                pts = obj_verts[0, active]
                return (pts * w.unsqueeze(-1)).sum(dim=0) / w.sum().clamp(min=1e-6)
            if mask.any():
                return obj_verts[0, mask].mean(dim=0)
            return ref_centroids_sample[finger].to(device=device, dtype=dtype)

        for finger in reference_finger_names:
            if finger not in ref_centroids_sample:
                continue
            centroid = _finger_centroid_from_ho(finger)
            ref_anchor_terms.append(
                ((centroid - ref_centroids_sample[finger].to(device=device, dtype=dtype))
                 / max(anchor_scale, 1e-4)).pow(2).mean()
            )

        for finger in new_finger_names:
            pid = _dip_part_id(finger)
            mask = part_ids == pid
            if mask.any():
                coverage_terms.append(F.relu(coverage_thresh - cmap[mask].mean()))
            else:
                coverage_terms.append(cmap.new_tensor(1.0))
            if finger in ref_centroids_sample:
                active = mask & (cmap > coverage_thresh * 0.5)
                if active.any():
                    w = cmap[active]
                    pts = obj_verts[0, active]
                    centroid = (pts * w.unsqueeze(-1)).sum(dim=0) / w.sum().clamp(min=1e-6)
                elif mask.any():
                    centroid = obj_verts[0, mask].mean(dim=0)
                else:
                    centroid = ref_centroids_sample[finger].to(device=device, dtype=dtype)
                anchor_terms.append(
                    ((centroid - ref_centroids_sample[finger].to(device=device, dtype=dtype))
                     / max(anchor_scale, 1e-4)).pow(2).mean()
                )

        reach_norm = _target_finger_mesh_reach_loss(
            hand_verts,
            obj_verts,
            hand_part_label,
            new_finger_names,
            min_surface_dist=min_surface_dist,
            max_surface_dist=max_surface_dist,
        )
        mesh_pene_norm = _target_finger_tip_penetration_loss(
            model,
            mano_layer,
            hand_verts,
            hand_part_label,
            new_finger_names,
            global_pose,
            mano_pose,
            mano_shape,
            mano_trans,
            clearance_floor=0.0,
        )
        coverage_norm = _clamp_unit(torch.stack(coverage_terms).mean()) if coverage_terms else reach_norm.new_zeros(())
        anchor_norm = _clamp_unit(torch.stack(anchor_terms).mean()) if anchor_terms else reach_norm.new_zeros(())
        ref_anchor_norm = (
            _clamp_unit(torch.stack(ref_anchor_terms).mean())
            if ref_anchor_terms
            else reach_norm.new_zeros(())
        )
        vf_norm = reach_norm.new_zeros(())
        if (
            vf_axis_origin is not None
            and vf_axis_ref_endpoint is not None
            and target_fingers
        ):
            thumb_centroids = _finger_contact_centroids(
                ho["contacts_object"], ho["partition_object"], obj_verts, ("thumb",),
                contact_thresh=coverage_thresh,
            )
            thumb_achieved = thumb_centroids["thumb"]
            vf_centroids = [_finger_centroid_from_ho(f) for f in target_fingers]
            vf_achieved = torch.stack(vf_centroids, dim=0).mean(dim=0)
            thumb_axis_norm = _clamp_unit(
                (thumb_achieved - vf_axis_origin).pow(2).sum() / max(vf_scale, 1e-4) ** 2
            )
            vf_ref_norm = _clamp_unit(
                (vf_achieved - vf_axis_ref_endpoint).pow(2).sum() / max(vf_scale, 1e-4) ** 2
            )
            vf_norm = (thumb_axis_norm + vf_ref_norm) / 2.0
        wrist_norm = _clamp_unit(
            (global_pose - g0).pow(2).sum() / max(wrist_rot_scale, 1e-4) ** 2
            + (mano_trans - t0).pow(2).sum() / max(wrist_trans_scale, 1e-4) ** 2
        )

        pred_p_full, _ = _sdf_forward(
            model, mano_layer, obj_verts, global_pose, mano_pose, mano_shape, mano_trans,
            zero_global=False,
        )
        loss_pene = _penetration_loss(
            pred_p_full, contact_part_tensor, eps, w_pene, dtype, device,
        )
        loss = (
            _weighted_term(w_reach, reach_norm)
            + _weighted_term(w_coverage, coverage_norm)
            + _weighted_term(w_anchor, anchor_norm)
            + _weighted_term(w_ref_anchor, ref_anchor_norm)
            + _weighted_term(w_vf, vf_norm)
            + _weighted_term(w_wrist, wrist_norm)
            + _weighted_term(w_mesh_pene, mesh_pene_norm)
            + loss_pene
        )
        optimizer.zero_grad()
        loss.backward()
        _apply_mano_pose_grad_mask(mano_pose, new_dof_mask)
        optimizer.step()
        _project_synthesis_wrist_thumb(
            global_pose, mano_trans, mano_pose, g0, t0, thumb0,
            wrist_rot_limit=wrist_rot_limit,
            wrist_trans_limit=wrist_trans_limit,
            thumb_pose_limit=thumb_pose_limit,
            thumb_lo=thumb_lo, thumb_hi=thumb_hi,
        )
        if ref_pose_pin is not None and ref_dof_t is not None:
            with torch.no_grad():
                mano_pose[:, ref_dof_t] = ref_pose_pin
        if limit_dof_t is not None:
            _project_pose_joint_limits(mano_pose, limit_dof_t, limit_lo_t, limit_hi_t)
        if it % 50 == 0 or it == reach_iter - 1:
            reach_mm = float(reach_norm.item()) * 1000.0
            print(
                f"phase1-alt reach iter {it} | loss {loss.item():.4f} "
                f"(reach {reach_mm:.2f} mm, mesh_pene {mesh_pene_norm.item():.3f}, "
                f"cov {coverage_norm.item():.3f})"
            )

    return (
        global_pose.detach(),
        mano_pose.detach(),
        mano_shape.detach(),
        mano_trans.detach(),
    )


def optimize_grasp_edit_phase2(*args, **kwargs):
    """Deprecated: use optimize_grasp_variant via sample_grasp_variants_sequential."""
    raise NotImplementedError(
        "optimize_grasp_edit_phase2 was replaced by optimize_grasp_variant; "
        "use ref_grasp_edit.sample_grasp_variants_sequential()."
    )


# Backward-compatible aliases (deprecated names from early "phase 0" terminology).
optimize_synthesis_phase0 = optimize_grasp_edit_phase1_alt
optimize_synthesis_reach_new_fingers = optimize_grasp_edit_phase1_alt_reach