"""
    build(::Type{MyClassicalHopfieldNetworkModel}, data::NamedTuple)

Builds a classical Hopfield network using Hebbian learning rule.

# Arguments
- `data::NamedTuple` with field `memories` - Array where each column is a binary memory pattern

# Returns
- `model::MyClassicalHopfieldNetworkModel` with encoded weights and energies
"""
function build(::Type{MyClassicalHopfieldNetworkModel}, data::NamedTuple)
    
    # Extract memories (should be N×K array, where N = number of neurons, K = number of patterns)
    memories = data.memories
    N, K = size(memories)
    
    # Compute weights using Hebbian learning rule: W = (1/K) * Σ(s_i ⊗ s_i^T)
    W = zeros(Float32, N, N)
    for k = 1:K
        s = Float32.(memories[:, k])  # get k-th memory pattern
        W .+= s * s'  # outer product
    end
    W ./= K  # normalize by number of patterns
    
    # Set diagonal to zero (no self-connections in Hopfield networks)
    for i = 1:N
        W[i, i] = 0.0
    end
    
    # Initialize bias to zero
    b = zeros(Float32, N)
    
    # Compute energy of each stored pattern
    energy_dict = Dict{Int64, Float32}()
    for k = 1:K
        s = Float32.(memories[:, k])
        E = -0.5 * (s' * W * s)[1] - (b' * s)[1]  # energy formula
        energy_dict[k] = E
    end
    
    return MyClassicalHopfieldNetworkModel(W, b, energy_dict)
end
