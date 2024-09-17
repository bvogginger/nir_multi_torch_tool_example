from typing import Optional

import nir
import torch
import brevitas.nn as qnn



def _extract_brevitas_module(module: torch.nn.Module) -> Optional[nir.NIRNode]:
    """Convert a single brevitas module to the equivalent object in the Neuromorphic
    Intermediate Representation (NIR). This function is used internally by the export_to_nir
    function to convert each submodule/layer of the network to the NIR.

    Currently supported brevitas modules: QuantLinear

    :param module: brevitas module
    :type module: torch.nn.Module

    :return: return the NIR node
    :rtype: Optional[nir.NIRNode]
    """
    if isinstance(module, qnn.QuantLinear):
        if module.bias is None:
            metadata = dict()
            # nir.Linear does not support any metadata right now
            if module.is_weight_quant_enabled:
                metadata["quant_weight"] = True
                metadata["quant_weight_scale"] = module.quant_weight_scale().detach().numpy()
                metadata["quant_weight_zero_point"] = module.quant_weight_zero_point().detach().numpy()
            return nir.Linear(
                weight=module.weight.data.detach().numpy(),
                metadata=metadata,
            )
        else:
            metadata = dict()
            if module.is_weight_quant_enabled:
                metadata["quant_weight"] = True
                metadata["quant_weight_scale"] = module.quant_weight_scale().detach().numpy()
                metadata["quant_weight_zero_point"] = module.quant_weight_zero_point().detach().numpy()
            if module.is_bias_quant_enabled:
                metadata["quant_bias"] = True
                metadata["quant_bias_scale"] = module.quant_bias_scale().detach().numpy()
                metadata["quant_bias_zero_point"] = module.quant_bias_zero_point().detach().numpy()

            return nir.Affine(
                weight=module.weight.data.detach().numpy(),
                bias=module.bias.data.detach().numpy(),
                metadata=metadata,
            )
    else:
        return None
