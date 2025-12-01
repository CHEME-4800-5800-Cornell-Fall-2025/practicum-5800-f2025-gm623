"""
    MyClassicalHopfieldNetworkModel

A type representing a classical Hopfield network with stored memories.

# Fields
- `W::Array{Float32, 2}`: Weight matrix (N×N) encoding the memories via Hebbian learning
- `b::Array{Float32, 1}`: Bias vector (N×1), typically set to zero
- `energy::Dict{Int64, Float32}`: Dictionary mapping memory indices to their network energy values
"""
mutable struct MyClassicalHopfieldNetworkModel
    W::Array{Float32, 2}  # weight matrix
    b::Array{Float32, 1}  # bias vector
    energy::Dict{Int64, Float32}  # energy of stored patterns
end