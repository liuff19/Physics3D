import sys

sys.path.append("gaussian-splatting")

import argparse
import math
import cv2
import torch
from torch import nn
import torch.nn.functional as F
import os
import numpy as np
import json
from tqdm import tqdm
from omegaconf import OmegaConf

# Gaussian splatting dependencies
from utils.sh_utils import eval_sh
from scene.gaussian_model import GaussianModel
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.cameras import Camera as GSCamera
from gaussian_renderer import render, GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import focal2fov

# MPM dependencies
from mpm_solver_warp.engine_utils import *
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
from mpm_solver_warp.mpm_utils import sum_array, sum_mat33, sum_vec3, wp_clamp, update_param
import warp as wp

# Particle filling dependencies
from particle_filling.filling import *

# Utils
from utils.decode_param import *
from utils.transformation_utils import *
from utils.camera_view_utils import *
from utils.render_utils import *
from utils.save_video import save_video
from utils.threestudio_utils import cleanup

from video_distillation.guidance import ModelscopeGuidance
from video_distillation.prompt_processors import ModelscopePromptProcessor


wp.init()
wp.config.verify_cuda = True

ti.init(arch=ti.cuda, device_memory_GB=8.0)


class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def load_checkpoint(model_path, iteration=-1):
    # Find checkpoint
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(
        checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
    )
    
    # sh_degree=0, if you use a 3D asset without spherical harmonics
    from plyfile import PlyData
    plydata = PlyData.read(checkpt_path)
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    
    # Load guassians
    sh_degree = int(math.sqrt((len(extra_f_names)+3) // 3)) - 1
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--physics_config", type=str, required=True)
    parser.add_argument("--guidance_config", type=str, default="./config/guidance/guidance.yaml")
    parser.add_argument("--white_bg", type=bool, default=True)
    parser.add_argument("--output_ply", action="store_true")
    parser.add_argument("--output_h5", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        AssertionError("Model path does not exist!")
    if not os.path.exists(args.physics_config):
        AssertionError("Scene config does not exist!")
    if not os.path.exists(args.guidance_config):
        AssertionError("Scene config does not exist!")
    if args.output_path is not None and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # load scene config
    print("Loading scene config...")
    (
        material_params,
        bc_params,
        time_params,
        preprocessing_params,
        camera_params,
    ) = decode_param_json(args.physics_config)

    # load gaussians
    print("Loading gaussians...")
    model_path = args.model_path
    gaussians = load_checkpoint(model_path)
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        if args.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    )

    # init the scene
    print("Initializing scene and pre-processing...")
    params = load_params_from_gs(gaussians, pipeline)

    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_screen_points = params["screen_points"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]

    # throw away low opacity kernels
    mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
    init_pos = init_pos[mask, :]
    init_cov = init_cov[mask, :]
    init_opacity = init_opacity[mask, :]
    init_screen_points = init_screen_points[mask, :]
    init_shs = init_shs[mask, :]
    
    # optimize moving parts only
    unselected_pos, unselected_cov, unselected_opacity, unselected_shs = (
        None,
        None,
        None,
        None,
    )
    moving_pts_path = os.path.join(model_path, "moving_part_points.ply")
    if os.path.exists(moving_pts_path):
        import point_cloud_utils as pcu
        moving_pts = pcu.load_mesh_v(moving_pts_path)
        moving_pts = torch.from_numpy(moving_pts).float().to("cuda")
        # moving_pts = apply_rotations(moving_pts, rotation_matrices)
        freeze_mask = find_far_points(
            init_pos, moving_pts, thres=0.05
        ).bool()
        moving_pts.to("cpu")
        unselected_pos = init_pos[freeze_mask, :]
        unselected_cov = init_cov[freeze_mask, :]
        unselected_opacity = init_opacity[freeze_mask, :]
        unselected_shs = init_shs[freeze_mask, :]

        init_pos = init_pos[~freeze_mask, :]
        init_cov = init_cov[~freeze_mask, :]
        init_opacity = init_opacity[~freeze_mask, :]
        init_shs = init_shs[~freeze_mask, :]

    # rorate and translate object
    if args.debug:
        if not os.path.exists("./log"):
            os.makedirs("./log")
        particle_position_tensor_to_ply(
            init_pos,
            "./log/init_particles.ply",
        )
    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"],
    )
    rotated_pos = apply_rotations(init_pos, rotation_matrices)

    if args.debug:
        particle_position_tensor_to_ply(rotated_pos, "./log/rotated_particles.ply")

    # select a sim area and save params of unslected particles
    if preprocessing_params["sim_area"] is not None:
        boundary = preprocessing_params["sim_area"]
        assert len(boundary) == 6
        mask = torch.ones(rotated_pos.shape[0], dtype=torch.bool).to(device="cuda")
        for i in range(3):
            mask = torch.logical_and(mask, rotated_pos[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, rotated_pos[:, i] < boundary[2 * i + 1])

        unselected_pos = init_pos[~mask, :]
        unselected_cov = init_cov[~mask, :]
        unselected_opacity = init_opacity[~mask, :]
        unselected_shs = init_shs[~mask, :]

        rotated_pos = rotated_pos[mask, :]
        init_cov = init_cov[mask, :]
        init_opacity = init_opacity[mask, :]
        init_shs = init_shs[mask, :]

    transformed_pos, scale_origin, original_mean_pos = transform2origin(rotated_pos)
    transformed_pos = shift2center111(transformed_pos)

    # modify covariance matrix accordingly
    init_cov = apply_cov_rotations(init_cov, rotation_matrices)
    init_cov = scale_origin * scale_origin * init_cov

    if args.debug:
        particle_position_tensor_to_ply(
            transformed_pos,
            "./log/transformed_particles.ply",
        )

    # fill particles if needed
    gs_num = transformed_pos.shape[0]
    device = "cuda:0"
    filling_params = preprocessing_params["particle_filling"]

    if filling_params is not None:
        print("Filling internal particles...")
        mpm_init_pos = fill_particles(
            pos=transformed_pos,
            opacity=init_opacity,
            cov=init_cov,
            grid_n=filling_params["n_grid"],
            max_samples=filling_params["max_particles_num"],
            grid_dx=material_params["grid_lim"] / filling_params["n_grid"],
            density_thres=filling_params["density_threshold"],
            search_thres=filling_params["search_threshold"],
            max_particles_per_cell=filling_params["max_partciels_per_cell"],
            search_exclude_dir=filling_params["search_exclude_direction"],
            ray_cast_dir=filling_params["ray_cast_direction"],
            boundary=filling_params["boundary"],
            smooth=filling_params["smooth"],
        ).to(device=device)

        if args.debug:
            particle_position_tensor_to_ply(mpm_init_pos, "./log/filled_particles.ply")
    else:
        mpm_init_pos = transformed_pos.to(device=device)

    # densify for high-frequency elastic objects
    init_len = mpm_init_pos.shape[0]
    # new_pts = []
    # for pt in mpm_init_pos:
    #     if pt[2] < 1.4 and pt[2] > 0.6:
    #         new_pts.append([pt[0]+0.05, pt[1], pt[2]])
    #         new_pts.append([pt[0]-0.05, pt[1], pt[2]])
    #         new_pts.append([pt[0], pt[1]+0.05, pt[2]])
    #         new_pts.append([pt[0], pt[1]-0.05, pt[2]])
    #         new_pts.append([pt[0]+0.05, pt[1]+0.05, pt[2]])
    #         new_pts.append([pt[0]+0.1, pt[1]-0.1, pt[2]])
    #         new_pts.append([pt[0]-0.1, pt[1]+0.1, pt[2]])
    #         new_pts.append([pt[0]-0.05, pt[1]-0.05, pt[2]])
    # mpm_init_pos = torch.concat([mpm_init_pos, torch.tensor(new_pts).to(device)]).to(torch.float32)

    # init the mpm solver
    print("Initializing MPM solver and setting up boundary conditions...")
    mpm_init_vol = get_particle_volume(
        mpm_init_pos,
        material_params["n_grid"],
        material_params["grid_lim"] / material_params["n_grid"],
        unifrom=material_params["material"] == "sand",
    ).to(device=device)

    if filling_params is not None and filling_params["visualize"] == True:
        shs, opacity, mpm_init_cov = init_filled_particles(
            mpm_init_pos[:gs_num],
            init_shs,
            init_cov,
            init_opacity,
            mpm_init_pos[gs_num:],
        )
        _pos = apply_inverse_rotations(
                undotransform2origin(
                    undoshift2center111(mpm_init_pos[gs_num:]), scale_origin, original_mean_pos
                ),
                rotation_matrices,
            )
        print(gaussians._xyz.shape)
        gaussians._xyz = nn.Parameter(torch.tensor(torch.cat([gaussians._xyz, _pos], 0), dtype=torch.float, device="cuda").requires_grad_(True))
        _features_dc = torch.zeros((_pos.shape[0], 1, 3)).to("cuda:0")
        print(gaussians._features_dc.shape)
        gaussians._features_dc = nn.Parameter(torch.tensor(torch.cat([gaussians._features_dc, _features_dc], 0), dtype=torch.float, device="cuda").requires_grad_(True))
        _features_rest = torch.zeros((_pos.shape[0], 15, 3)).to("cuda:0")
        print(gaussians._features_rest.shape)
        gaussians._features_rest = nn.Parameter(torch.tensor(torch.cat([gaussians._features_rest, _features_rest], 0), dtype=torch.float, device="cuda").requires_grad_(True))
        _opacity = torch.zeros((_pos.shape[0], 1)).to("cuda:0")
        gaussians._opacity = nn.Parameter(torch.tensor(torch.cat([gaussians._opacity, _opacity], 0), dtype=torch.float, device="cuda").requires_grad_(True))
        _scaling = torch.zeros((_pos.shape[0], 3)).to("cuda:0")
        gaussians._scaling = nn.Parameter(torch.tensor(torch.cat([gaussians._scaling, _scaling], 0), dtype=torch.float, device="cuda").requires_grad_(True))
        _rotation = torch.zeros((_pos.shape[0], 4)).to("cuda:0")
        gaussians._rotation = nn.Parameter(torch.tensor(torch.cat([gaussians._rotation, _rotation], 0), dtype=torch.float, device="cuda").requires_grad_(True))

        gs_num = mpm_init_pos.shape[0]
    else:
        mpm_init_cov = torch.zeros((mpm_init_pos.shape[0], 6), device=device)
        mpm_init_cov[:gs_num] = init_cov
        shs = init_shs
        opacity = init_opacity

    if args.debug:
        print("check *.ply files to see if it's ready for simulation")

    # set up the mpm solver
    mpm_solver = MPM_Simulator_WARP(10)
    mpm_solver.load_initial_data_from_torch(
        mpm_init_pos,
        mpm_init_vol,
        mpm_init_cov,
        n_grid=material_params["n_grid"],
        grid_lim=material_params["grid_lim"],
    )
    mpm_solver.set_parameters_dict(material_params)

    # Note: boundary conditions may depend on mass, so the order cannot be changed!
    set_boundary_conditions(mpm_solver, bc_params, time_params)
    
    # moving_pts_path = os.path.join(model_path, "moving_part_points.ply")
    # if os.path.exists(moving_pts_path):
    #     import point_cloud_utils as pcu
    #     moving_pts = pcu.load_mesh_v(moving_pts_path)
    #     moving_pts = torch.from_numpy(moving_pts).float().to("cuda")
    #     moving_pts = apply_rotations(moving_pts, rotation_matrices)
    #     moving_pts, moving_scale_origin, moving_original_mean_pos = transform2origin(moving_pts)
    #     moving_pts = shift2center111(moving_pts)
    #     get_particle_volume(
    #         moving_pts,
    #         material_params["n_grid"],
    #         material_params["grid_lim"] / material_params["n_grid"],
    #         unifrom=False,
    #     )
    #     freeze_mask = find_far_points(
    #         mpm_init_pos, moving_pts, thres=0.5
    #     ).bool()
    #     freeze_pts = mpm_init_pos[freeze_mask, :]
    #     apply_grid_bc_w_freeze_pts(
    #         mpm_solver.mpm_model.n_grid, mpm_solver.mpm_model.grid_lim, freeze_pts, mpm_solver
    #     )
    
    tape = wp.Tape()

    # mpm_solver.finalize_mu_lam()

    # camera setting
    mpm_space_viewpoint_center = (
        torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    )
    mpm_space_vertical_upward_axis = (
        torch.tensor(camera_params["mpm_space_vertical_upward_axis"])
        .reshape((1, 3))
        .cuda()
    )
    (
        viewpoint_center_worldspace,
        observant_coordinates,
    ) = get_center_view_worldspace_and_observant_coordinate(
        mpm_space_viewpoint_center,
        mpm_space_vertical_upward_axis,
        rotation_matrices,
        scale_origin,
        original_mean_pos,
    )

    # run the simulation
    if args.output_ply or args.output_h5:
        directory_to_save = os.path.join(args.output_path, "simulation_ply")
        if not os.path.exists(directory_to_save):
            os.makedirs(directory_to_save)

        save_data_at_frame(
            mpm_solver,
            directory_to_save,
            0,
            save_to_ply=args.output_ply,
            save_to_h5=args.output_h5,
        )

    substep_dt = time_params["substep_dt"]
    frame_dt = time_params["frame_dt"]
    frame_num = time_params["frame_num"]
    step_per_frame = int(frame_dt / substep_dt)
    opacity_render = opacity
    shs_render = shs
    height = None
    width = None
    
    yaml_confs = OmegaConf.load(args.guidance_config)
    yaml_confs.prompt_processor.prompt = args.prompt
    guidance = ModelscopeGuidance(yaml_confs.guidance)
    prompt_processor = ModelscopePromptProcessor(yaml_confs.prompt_processor)
    prompt_utils = prompt_processor()
    
    stage_num = 8
    frame_per_stage = 16
    for batch in range(50):
        loss_value = 0.
        img_list = []
        tape.reset()
        with tape:
            mpm_solver.finalize_mu_lam()
        
        for _ in range(step_per_frame * (batch % stage_num)):
            mpm_solver.p2g2p(None, substep_dt, device=device)
        
        for frame in tqdm(range(frame_per_stage)):
            current_camera = get_camera_view(
                model_path,
                default_camera_index=camera_params["default_camera_index"],
                center_view_world_space=viewpoint_center_worldspace,
                observant_coordinates=observant_coordinates,
                show_hint=camera_params["show_hint"],
                init_azimuthm=camera_params["init_azimuthm"],
                init_elevation=camera_params["init_elevation"],
                init_radius=camera_params["init_radius"],
                move_camera=camera_params["move_camera"],
                current_frame=frame,
                delta_a=camera_params["delta_a"],
                delta_e=camera_params["delta_e"],
                delta_r=camera_params["delta_r"],
            )
            rasterize = initialize_resterize(
                current_camera, gaussians, pipeline, background
            )
            
            for _ in range(step_per_frame * (1 + stage_num) - 1):
                mpm_solver.p2g2p(frame, substep_dt, device=device)
            with tape:
                mpm_solver.p2g2p(frame, substep_dt, device=device)

                pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
                cov3D = mpm_solver.export_particle_cov_to_torch()
                rot = mpm_solver.export_particle_R_to_torch()
            
            cov3D = cov3D.view(-1, 6)[:gs_num].to(device)
            rot = rot.view(-1, 3, 3)[:gs_num].to(device)

            pos = pos[:init_len,:]
            pos = apply_inverse_rotations(
                undotransform2origin(
                    undoshift2center111(pos), scale_origin, original_mean_pos
                ),
                rotation_matrices,
            )
            cov3D = cov3D / (scale_origin * scale_origin)
            cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
            opacity = opacity_render
            shs = shs_render
            if preprocessing_params["sim_area"] is not None:
                pos = torch.cat([pos, unselected_pos], dim=0)
                cov3D = torch.cat([cov3D, unselected_cov], dim=0)
                opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
                shs = torch.cat([shs_render, unselected_shs], dim=0)
            if os.path.exists(moving_pts_path):
                pos = torch.cat([pos, unselected_pos], dim=0)
                cov3D = torch.cat([cov3D, unselected_cov], dim=0)
                opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
                shs = torch.cat([shs_render, unselected_shs], dim=0)

            colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)
            rendering, raddi = rasterize(
                means3D=pos,
                means2D=pos,
                shs=None,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D,
            )
            img_list.append(rendering)
        
        loss = 0.
        img_list = torch.stack(img_list)
        guidance_out = guidance(img_list, prompt_utils, torch.Tensor([camera_params['init_elevation']]), torch.Tensor([camera_params['init_azimuthm']]), torch.Tensor([camera_params['init_radius']]), rgb_as_latents=False, num_frames=frame_per_stage, train_dynamic_camera=False)
        print(guidance_out)
        for name, value in guidance_out.items():
            if name.startswith('loss_'):
                loss += value * 3e-4
        loss = loss / stage_num
        print(loss)
        loss.backward(retain_graph=True)
        loss_value += loss.item()
        grad_x = mpm_solver.mpm_state.particle_x.grad
        grad_cov = mpm_solver.mpm_state.particle_cov.grad
        grad_r = mpm_solver.mpm_state.particle_R.grad
        loss_wp = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        print(torch.max(wp.to_torch(grad_x)), torch.mean(wp.to_torch(grad_x)))
        wp.launch(sum_vec3, mpm_solver.n_particles, [mpm_solver.mpm_state.particle_x, grad_x], [loss_wp], device=device)
        wp.launch(sum_array, mpm_solver.n_particles*6, [mpm_solver.mpm_state.particle_cov, grad_cov], [loss_wp], device=device)
        wp.launch(sum_mat33, mpm_solver.n_particles, [mpm_solver.mpm_state.particle_R, grad_r], [loss_wp], device=device)
        tape.backward(loss=loss_wp)
        grad = wp.to_torch(mpm_solver.mpm_model.E.grad)
        max_grad, min_grad = torch.max(grad), torch.min(grad)
        grad = (grad - min_grad) / (max_grad - min_grad) - 0.5 if max_grad - min_grad != 0 else torch.zeros_like(grad)
        wp.launch(update_param, mpm_solver.n_particles, [mpm_solver.mpm_model.E, wp.from_torch(grad), 1.0, 2])
        
        # add
        grad_mu_N = wp.to_torch(mpm_solver.mpm_model.mu_N.grad)
        max_grad, min_grad = torch.max(grad_mu_N), torch.min(grad_mu_N)
        grad_mu_N = (grad_mu_N - min_grad) / (max_grad - min_grad) - 0.5 if max_grad - min_grad != 0 else torch.zeros_like(grad_mu_N)
        wp.launch(update_param, mpm_solver.n_particles, [mpm_solver.mpm_model.mu_N, wp.from_torch(grad_mu_N), 1.0, 2.0])
        
        grad_lam_N = wp.to_torch(mpm_solver.mpm_model.lam_N.grad)
        max_grad, min_grad = torch.max(grad_lam_N), torch.min(grad_lam_N)
        grad_lam_N = (grad_lam_N - min_grad) / (max_grad - min_grad) - 0.5 if max_grad - min_grad != 0 else torch.zeros_like(grad_lam_N)
        wp.launch(update_param, mpm_solver.n_particles, [mpm_solver.mpm_model.lam_N, wp.from_torch(grad_lam_N), 1.0, 2.0])
        
        grad_viscosity = wp.to_torch(mpm_solver.mpm_model.viscosity.grad)
        max_grad, min_grad = torch.max(grad_viscosity), torch.min(grad_viscosity)
        grad_viscosity = (grad_viscosity - min_grad) / (max_grad - min_grad) - 0.5 if max_grad - min_grad != 0 else torch.zeros_like(grad_viscosity)
        wp.launch(update_param, mpm_solver.n_particles, [mpm_solver.mpm_model.viscosity, wp.from_torch(grad_viscosity), 1.0, 3.0])
        
        print("grad: ", torch.mean(grad), torch.max(wp.to_torch(mpm_solver.mpm_model.E.grad)), torch.min(wp.to_torch(mpm_solver.mpm_model.E.grad)))
        print("grad_mu_N: ", torch.mean(grad_mu_N), torch.max(wp.to_torch(mpm_solver.mpm_model.mu_N.grad)), torch.min(wp.to_torch(mpm_solver.mpm_model.mu_N.grad)))
        print("grad_lam_N: ", torch.mean(grad_lam_N))
        print("grad_viscosity: ", torch.mean(grad_viscosity))
        print("E: ", torch.max(wp.to_torch(mpm_solver.mpm_model.E)), torch.min(wp.to_torch(mpm_solver.mpm_model.E)), torch.mean(wp.to_torch(mpm_solver.mpm_model.E)))
        print("mu_N: ", torch.max(wp.to_torch(mpm_solver.mpm_model.mu_N)), torch.min(wp.to_torch(mpm_solver.mpm_model.mu_N)), torch.mean(wp.to_torch(mpm_solver.mpm_model.mu_N)))
        print("lam_N: ", torch.max(wp.to_torch(mpm_solver.mpm_model.lam_N)), torch.min(wp.to_torch(mpm_solver.mpm_model.lam_N)), torch.mean(wp.to_torch(mpm_solver.mpm_model.lam_N)))
        print("viscosity: ", torch.max(wp.to_torch(mpm_solver.mpm_model.viscosity)), torch.min(wp.to_torch(mpm_solver.mpm_model.viscosity)), torch.mean(wp.to_torch(mpm_solver.mpm_model.viscosity)))
        
        mpm_solver.reset_pos_from_torch(mpm_init_pos, mpm_init_vol, mpm_init_cov)
        if batch % 2 == 0:
            mpm_solver.finalize_mu_lam()
            for frame in tqdm(range(stage_num * frame_per_stage)):
                current_camera = get_camera_view(
                    model_path,
                    default_camera_index=camera_params["default_camera_index"],
                    center_view_world_space=viewpoint_center_worldspace,
                    observant_coordinates=observant_coordinates,
                    show_hint=camera_params["show_hint"],
                    init_azimuthm=camera_params["init_azimuthm"],
                    init_elevation=camera_params["init_elevation"],
                    init_radius=camera_params["init_radius"],
                    move_camera=camera_params["move_camera"],
                    current_frame=frame,
                    delta_a=camera_params["delta_a"],
                    delta_e=camera_params["delta_e"],
                    delta_r=camera_params["delta_r"],
                )
                rasterize = initialize_resterize(
                    current_camera, gaussians, pipeline, background
                )
                
                for _ in range(step_per_frame):
                    mpm_solver.p2g2p(frame, substep_dt, device=device)

                pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
                cov3D = mpm_solver.export_particle_cov_to_torch()
                rot = mpm_solver.export_particle_R_to_torch()
                
                cov3D = cov3D.view(-1, 6)[:gs_num].to(device)
                rot = rot.view(-1, 3, 3)[:gs_num].to(device)

                pos = pos[:init_len,:]
                pos = apply_inverse_rotations(
                    undotransform2origin(
                        undoshift2center111(pos), scale_origin, original_mean_pos
                    ),
                    rotation_matrices,
                )
                cov3D = cov3D / (scale_origin * scale_origin)
                cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
                opacity = opacity_render
                shs = shs_render
                if preprocessing_params["sim_area"] is not None:
                    pos = torch.cat([pos, unselected_pos], dim=0)
                    cov3D = torch.cat([cov3D, unselected_cov], dim=0)
                    opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
                    shs = torch.cat([shs_render, unselected_shs], dim=0)
                if os.path.exists(moving_pts_path):
                    pos = torch.cat([pos, unselected_pos], dim=0)
                    cov3D = torch.cat([cov3D, unselected_cov], dim=0)
                    opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
                    shs = torch.cat([shs_render, unselected_shs], dim=0)

                colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)
                rendering, raddi = rasterize(
                    means3D=pos,
                    means2D=init_screen_points,
                    shs=None,
                    colors_precomp=colors_precomp,
                    opacities=opacity,
                    scales=None,
                    rotations=None,
                    cov3D_precomp=cov3D,
                )
                
                cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
                cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                if height is None or width is None:
                    height = cv2_img.shape[0] // 2 * 2
                    width = cv2_img.shape[1] // 2 * 2
                assert args.output_path is not None
                cv2.imwrite(
                    os.path.join(args.output_path, f"{frame}.png".rjust(8, "0")),
                    255 * cv2_img,
                )
            save_video(args.output_path, os.path.join(args.output_path, 'video%02d.mp4' % batch))
