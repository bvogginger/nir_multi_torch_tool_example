from typing import Optional

import numpy as np

import torch
import torch.nn as nn
import snntorch
import snntorch as snn
from snntorch.export_nir import _extract_snntorch_module
import brevitas
import brevitas.nn as qnn
import nir
import nirtorch

from brevitas_to_nir import _extract_brevitas_module

n_in = 10
n_hidden_1 = 100
n_hidden_2 = 50
n_out = 2

# Example SNN with QuantLinear layers
net = nn.Sequential(
        # Affine with quantized weights and quantized bias
        qnn.QuantLinear(n_in, n_hidden_1, bias=True, weight_bit_width=8, bias_quant=brevitas.quant.Int8BiasPerTensorFloatInternalScaling),
        snn.Leaky(beta=0.9, init_hidden=True),
        # Affine with quantized weights but non-quantized bias
        qnn.QuantLinear(n_hidden_1, n_hidden_2, bias=True, weight_bit_width=8),
        snn.Leaky(beta=0.9, init_hidden=True),
        # Linear with quantized weights (no bias)
        qnn.QuantLinear(n_hidden_2, n_out, bias=False, weight_bit_width=8),
        snn.Leaky(beta=0.9, init_hidden=True, output=True)
        )

net.to(device="cpu")

def combined_module_mapper(module: torch.nn.Module) -> Optional[nir.NIRNode]:
    """module mapper combining brevitas and snntorch modules"""
    node = _extract_brevitas_module(module)
    if node is None:
        node = _extract_snntorch_module(module)
    return node

sample_data = torch.randn(n_in)
print(sample_data)
nir_graph = nirtorch.extract_nir_graph(
    net,
    combined_module_mapper,
    sample_data,
    model_name="QuantizedSNN",
    ignore_submodules_of=[snn.RLeaky, snn.RSynaptic, qnn.QuantLinear],
)

from spinnaker2 import s2_nir
print("Summary original graph")
s2_nir.model_summary(nir_graph)
print("")

nir.write("qnn.nir", nir_graph)

print("Summary re-imported graph")
new_nir_graph = nir.read("qnn.nir")
s2_nir.model_summary(new_nir_graph)
print("")

# print metadata if available
for name, node in new_nir_graph.nodes.items():
    if hasattr(node, "metadata"):
        print(name, type(node), node.metadata)
    else:
        print(name, type(node))

# Compare brevitas integer weights with rescaled ones from NIR
l1_int_weights = net[0].int_weight().detach().numpy()
print(l1_int_weights)

# get scale and zero point of weights of first linear layer
l1_float_weights = new_nir_graph.nodes["0"].weight
md = new_nir_graph.nodes["0"].metadata
weight_scale = md["quant_weight_scale"]
weight_zero_point = md["quant_weight_zero_point"]
assert(weight_zero_point == 0.0)
scaled_weights = l1_float_weights/weight_scale # scale weights to integer range

print(l1_int_weights[:2])
print(scaled_weights.astype(np.int8)[:2])
print(scaled_weights.round().astype(np.int8)[:2])

# check that int weights are the same
assert(np.array_equal(scaled_weights.round().astype(np.int8), l1_int_weights))

print(new_nir_graph.edges)
