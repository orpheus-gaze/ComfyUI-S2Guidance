import comfy.model_patcher
import comfy.samplers
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

import torch
import numpy as np
import random
import math

class S2Guidance_DIT(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:        
        return io.Schema(
            node_id="S2Guidance_DIT",
            display_name=f"✨S2GuidanceDIT",
            category="✨ S2GUIDANCE",
            description=(
                "Enables S²-guidance for certain diffusion models which should lead to better prompt adherence. It achieves this by subtracting a subnetwork of layers from the guidance which should make it more robust than normal CFG-guidance."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("s2_guidance_scale",
                    default=0.25,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "The strength of the S² guidance scale. \n\n(no effect=0.0, strong effect=2.0, default=0.25)"
                    )
                ),                
                io.Int.Input("skip_layers_percentage",
                    default=1,
                    min=1,
                    max=100,
                    step=1,
                    optional=True,
                    tooltip=(
                        "The skip_layers_percentage variable dictates the percentage of how many layers out of the total that should be skipped. \n\n(one=1, all=100, default=1)"
                    )
                ),
            ],
            outputs=[
                io.Model.Output("model"),                
            ]
        )
    
    @classmethod
    def execute(cls, model, s2_guidance_scale: float, skip_layers_percentage: int) -> io.NodeOutput:
        '''
            This is an implementation of S²-guidance, as described in arxiv 2508.12880, and returns a patched model that applies the guidance during generation by subtracting the weighted (s2_guidance_scale) results of a subnetwork of the model, that is, the calculated predicted condition while layers are removed, from the CFG result.
            
            input: 
                - unpatched model
                - s2_guidance_scale
                - skip_layers_percentage
                
            returns: 
                - S²-guidance patched model
        '''      
        
        def skip (args, extra_args):
            # The function simply returns args without modification, which means the block's output equals its input—effectively skipping the block's computation entirely.
            return args
        
        def post_cfg_function(args):
            model = args["model"]
            
            def apply_s2_guidance(args, naming, double_layers: int, single_layers: int):
                cond_pred = args["cond_denoised"]
                uncond_pred = args["uncond_denoised"]
                cond = args["cond"]
                cfg_result = args["denoised"]
                sigma = args["sigma"]
                model_options = args["model_options"].copy()
                x = args["input"]
                
                total_num_blocks = double_layers + single_layers
            
                skip_layers_prob = skip_layers_percentage / 100.0
                skip_layers = int(math.ceil(skip_layers_prob * total_num_blocks)) # skip_layers_percentage is set as user input
                all_indices = list(range(total_num_blocks))                
                layers_to_skip = random.sample(all_indices, skip_layers)
                
                for layer in layers_to_skip:
                    drop_block_prob = random.randint(1, 100) # Random threshold to exceed for the dropout_odds_percentage. # CHANGELOG 20260510
                    block_selection = layer # Random block to drop out.
                    
                    # The dropout_odds_percentage variable dictates how often the blocks should be dropped. never=0, always=100, default=100. #CHANGELOG 20260510
                    if block_selection <= double_layers:
                        # Do S² guidance on double blocks
                        model_options = comfy.model_patcher.set_model_options_patch_replace(
                                                                        model_options, 
                                                                        skip, 
                                                                        "dit", 
                                                                        naming[0], 
                                                                        block_selection
                                                                    )
                    else:
                        # Do S² guidance on single blocks
                        block_selection = block_selection - double_layers # CHANGELOG 20260511
                        model_options = comfy.model_patcher.set_model_options_patch_replace(
                                                            model_options, 
                                                            skip, 
                                                            "dit", 
                                                            naming[1], 
                                                            block_selection
                                                        )
                                                                                                          
                (s2_cond_pred,) = comfy.samplers.calc_cond_batch(model, [cond], x, sigma, model_options)
                
                if args["model_options"] != model_options:
                    # if model options was changed subtract the s2 guidance
                    '''
                        To make sure that you only subtract the differences from the cfg result you remove the parallel component from the prediction thereby 
                        giving you only the orthogonal difference between the cfg prediction and the subnetwork prediction. 
                    '''
                    # The "Stable" version of Eq 19
                    refined = cfg_result - (s2_guidance_scale * s2_cond_pred)

                    # Match the 'energy' of the original CFG so colors stay correct
                    return refined * (torch.norm(cfg_result) / torch.norm(refined))
                    
                else: 
                    # if for any reason the result was unchanged return the cfg result
                    return cfg_result
            
            if model.model_config.unet_config.get("n_layers"): # for at least z-image
                double_layers = model.model_config.unet_config["n_layers"]
                single_layers = 0 # single block arch
                
                naming = ["layers", "layers"]
                
                return apply_s2_guidance(args, naming, double_layers, single_layers)
            
            if model.model_config.unet_config.get("depth"): # for at least flux 2 klein
                double_layers = model.model_config.unet_config["depth"]
                single_layers = model.model_config.unet_config["depth_single_blocks"]
                
                naming = ["double_blocks", "single_blocks"]
                
                return apply_s2_guidance(args, naming, double_layers, single_layers)
                
            if model.model_config.unet_config.get("n_double_layers"): # for auraflow type configuration of models
                double_layers = model.model_config.unet_config["n_double_layers"]
                n_layers = model.model_config.unet_config["n_layers"]
                single_layers = n_layers - double_layers
                
                naming = ["double_layers", "single_layers"]
                
                return apply_s2_guidance(args, naming, double_layers, single_layers)
                
            else: 
                print("Model not supported for S²-Guidance")
                return cfg_result # if not supported architecture do not change function
                
        print("Using S²-Guidance")
        m = model.clone()
        m.set_model_sampler_post_cfg_function(post_cfg_function)

        return io.NodeOutput(m)
        
        
class PerpoGuidance_DIT(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:        
        return io.Schema(
            node_id="Perpo-Guidance_DIT",
            display_name=f"✨Perpo-GuidanceDIT",
            category="✨ S2GUIDANCE",
            description=(
                "Enables Perpo-guidance for certain diffusion models which should lead to better prompt adherence. It achieves this by subtracting a subnetwork of layers from the guidance which should make it more robust than normal CFG-guidance."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("perpo_guidance_scale",
                    default=1.0,
                    min=0.0,
                    max=10.0,
                    step=0.1,
                    optional=True,
                    tooltip=(
                        "The strength of the perpo guidance scale. \n\n(no effect=0.0, strong effect=10.0, default=1)"
                    )
                ),                
                io.Int.Input("skip_layers_percentage",
                    default=1,
                    min=1,
                    max=100,
                    step=1,
                    optional=True,
                    tooltip=(
                        "The skip_layers_percentage variable dictates the percentage of how many layers out of the total that should be skipped. \n\n(one=1, all=100, default=1)"
                    )
                ),
            ],
            outputs=[
                io.Model.Output("model"),                
            ]
        )
    
    @classmethod
    def execute(cls, model, perpo_guidance_scale: float, skip_layers_percentage: int) -> io.NodeOutput:
        '''
            This is an modified implementation of S²-guidance, as described in arxiv 2508.12880, which I've named as Perpo-Guidance, 
            and returns a patched model that applies the guidance during generation by subtracting the weighted (perpo_guidance_scale) 
            results of a subnetwork of the model, that is, the calculated predicted condition while layers are removed, from the CFG result. 
            The difference from the original implementation (as far as I can tell) is that the original normalises the result for the 
            subtracted network, while here the perpendicular result, that is, the difference of the subnetwork prediction is subtracted 
            from the original cfg result.
            
            input: 
                - unpatched model
                - perpo_guidance_scale
                - skip_layers_percentage
                
            returns: 
                - S²-guidance patched model
        '''     
        
        def skip (args, extra_args):
            # The function simply returns args without modification, which means the block's output equals its input—effectively skipping the block's computation entirely.
            return args
        
        def post_cfg_function(args):
            model = args["model"]
            
            def apply_perpo_guidance(args, naming, double_layers: int, single_layers: int):
                cond_pred = args["cond_denoised"]
                uncond_pred = args["uncond_denoised"]
                cond = args["cond"]
                cfg_result = args["denoised"]
                sigma = args["sigma"]
                model_options = args["model_options"].copy()
                x = args["input"]
                
                total_num_blocks = double_layers + single_layers
            
                skip_layers_prob = skip_layers_percentage / 100.0
                skip_layers = int(math.ceil(skip_layers_prob * total_num_blocks)) # skip_layers_percentage is set as user input
                all_indices = list(range(total_num_blocks))                
                layers_to_skip = random.sample(all_indices, skip_layers)
                
                for layer in layers_to_skip:
                    drop_block_prob = random.randint(1, 100) # Random threshold to exceed for the dropout_odds_percentage. # CHANGELOG 20260510
                    block_selection = layer # Random block to drop out.
                    
                    # The dropout_odds_percentage variable dictates how often the blocks should be dropped. never=0, always=100, default=100. #CHANGELOG 20260510
                    if block_selection <= double_layers:
                        # Do Perpo-guidance on double blocks
                        model_options = comfy.model_patcher.set_model_options_patch_replace(
                                                                        model_options, 
                                                                        skip, 
                                                                        "dit", 
                                                                        naming[0], 
                                                                        block_selection
                                                                    )
                    else:
                        # Do Perpo-guidance on single blocks
                        block_selection = block_selection - double_layers # CHANGELOG 20260511
                        model_options = comfy.model_patcher.set_model_options_patch_replace(
                                                            model_options, 
                                                            skip, 
                                                            "dit", 
                                                            naming[1], 
                                                            block_selection
                                                        )
                                                                                                          
                (perpo_cond_pred,) = comfy.samplers.calc_cond_batch(model, [cond], x, sigma, model_options)
                
                if args["model_options"] != model_options:
                    # if model options was changed subtract the perpo-guidance
                    '''
                        If model options was changed subtract the perpo-guidance. To make sure that you only subtract the differences from the cfg 
                        result you remove the parallel component from the prediction thereby giving you only the orthogonal difference 
                        between the cfg prediction and the subnetwork prediction. 
                    '''
                    cfg_flat = cfg_result.flatten()
                    perpo_flat = perpo_cond_pred.flatten()

                    proj = torch.dot(perpo_flat, cfg_flat) / (torch.dot(cfg_flat, cfg_flat) + 1e-6)
                    parallel = proj * cfg_result
                    
                    # Subtracts the parallel part, leaving only the component that is perpendicular to cfg_result. 
                    # It isolates the pure difference that isn't already captured by CFG.
                    orthogonal = perpo_cond_pred - parallel
                    
                    return cfg_result - (perpo_guidance_scale * orthogonal)
                    
                else: 
                    # if for any reason the result was unchanged return the cfg result
                    return cfg_result
            
            if model.model_config.unet_config.get("n_layers"): # for at least z-image
                double_layers = model.model_config.unet_config["n_layers"]
                single_layers = 0 # single block arch
                
                naming = ["layers", "layers"]
                
                return apply_perpo_guidance(args, naming, double_layers, single_layers)
            
            if model.model_config.unet_config.get("depth"): # for at least flux 2 klein
                double_layers = model.model_config.unet_config["depth"]
                single_layers = model.model_config.unet_config["depth_single_blocks"]
                
                naming = ["double_blocks", "single_blocks"]
                
                return apply_perpo_guidance(args, naming, double_layers, single_layers)
                
            if model.model_config.unet_config.get("n_double_layers"): # for auraflow type configuration of models
                double_layers = model.model_config.unet_config["n_double_layers"]
                n_layers = model.model_config.unet_config["n_layers"]
                single_layers = n_layers - double_layers
                
                naming = ["double_layers", "single_layers"]
                
                return apply_perpo_guidance(args, naming, double_layers, single_layers)
                
            else: 
                print("Model not supported for Perpo-Guidance")
                return cfg_result # if not supported architecture do not change function
                
        
        print("Using Perpo-Guidance")
        m = model.clone()
        m.set_model_sampler_post_cfg_function(post_cfg_function)

        return io.NodeOutput(m)
        

class S2Guidance_DITExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            S2Guidance_DIT, PerpoGuidance_DIT, 
        ]


async def comfy_entrypoint() -> S2Guidance_DITExtension:
    return S2Guidance_DITExtension()
