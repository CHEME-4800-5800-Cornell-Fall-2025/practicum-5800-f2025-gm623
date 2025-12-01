"""
    decode(s::Array{Int32, 1})

Converts a flattened binary state vector (from 1D array) back to a 2D image matrix.
Assumes square images (length(s) = n²).

# Arguments
- `s::Array{Int32, 1}`: Binary state vector (-1 or +1 values)

# Returns
- `image::Array{Float32, 2}`: 2D image matrix (n×n)
"""
function decode(s::Array{Int32, 1})
    N = length(s)
    n = Int(sqrt(N))  # assume square image
    
    # Reshape the vector into a square matrix
    # Convert -1 to 0, and 1 to 1
    image = zeros(Float32, n, n)
    for i = 1:N
        row = div(i - 1, n) + 1
        col = mod(i - 1, n) + 1
        # Map -1 to 0.0 and 1 to 1.0
        image[row, col] = (s[i] == 1) ? 1.0 : 0.0
    end
    
    return image
end

"""
    hamming(s1::Array{Int32, 1}, s2::Array{Int32, 1})

Computes the Hamming distance between two binary vectors.

# Arguments
- `s1::Array{Int32, 1}`: First binary vector
- `s2::Array{Int32, 1}`: Second binary vector

# Returns
- `distance::Int64`: Number of positions where s1 and s2 differ
"""
function hamming(s1::Array{Int32, 1}, s2::Array{Int32, 1})
    return sum(s1 .!= s2)
end

"""
    recover(model::MyClassicalHopfieldNetworkModel, s₀::Array{Int32, 1}, 
            true_image_energy::Float32; maxiterations::Int64=1000, 
            patience::Union{Int, Nothing}=nothing, 
            miniterations_before_convergence::Union{Int, Nothing}=nothing)

Performs asynchronous memory retrieval from the Hopfield network.

# Arguments
- `model::MyClassicalHopfieldNetworkModel`: Encoded network model
- `s₀::Array{Int32, 1}`: Initial (corrupted) state vector
- `true_image_energy::Float32`: Energy of the target memory (for reference)
- `maxiterations::Int64`: Maximum number of update steps (default: 1000)
- `patience::Union{Int, Nothing}`: Number of consecutive identical states for convergence (default: 5)
- `miniterations_before_convergence::Union{Int, Nothing}`: Minimum iterations before checking convergence

# Returns
- `frames::Dict{Int64, Array{Int32, 1}}`: Dictionary mapping iteration to network state
- `energydictionary::Dict{Int64, Float32}`: Dictionary mapping iteration to network energy
"""
function recover(model::MyClassicalHopfieldNetworkModel, s₀::Array{Int32, 1}, 
                 true_image_energy::Float32; 
                 maxiterations::Int64=1000, 
                 patience::Union{Int, Nothing}=nothing,
                 miniterations_before_convergence::Union{Int, Nothing}=nothing)
    
    # Set default patience value
    if patience === nothing
        patience = 5
    end
    
    # Set minimum iterations before convergence check
    if miniterations_before_convergence === nothing
        miniterations_before_convergence = patience
    end
    
    N = length(s₀)
    W = model.W
    b = model.b
    
    # Initialize state and tracking variables
    s = copy(s₀)
    frames = Dict{Int64, Array{Int32, 1}}()
    energydictionary = Dict{Int64, Float32}()
    
    # Compute initial energy
    E₀ = -0.5 * Float32.(s)' * W * Float32.(s) .- b' * Float32.(s)
    if isa(E₀, Array)
        E₀ = E₀[1]
    end
    
    frames[0] = copy(s)
    energydictionary[0] = E₀
    
    # State history queue for convergence checking
    state_history = DataStructures.Queue{Array{Int32, 1}}()
    
    converged = false
    t = 1
    
    while !converged && t ≤ maxiterations
        # Store old state
        s_old = copy(s)
        
        # Randomly select a neuron to update (asynchronous update)
        i = rand(1:N)
        
        # Compute activation for neuron i
        activation = 0.0
        for j = 1:N
            if i ≠ j  # no self-connections
                activation += W[i, j] * s[j]
            end
        end
        activation -= b[i]
        
        # Update neuron state using sign function
        s[i] = sign(activation) > 0 ? Int32(1) : Int32(-1)
        
        # Store the state and compute energy
        frames[t] = copy(s)
        E = -0.5 * Float32.(s)' * W * Float32.(s) .- b' * Float32.(s)
        if isa(E, Array)
            E = E[1]
        end
        energydictionary[t] = E
        
        # Add current state to history
        enqueue!(state_history, copy(s))
        if length(state_history) > patience
            dequeue!(state_history)
        end
        
        # Check for convergence (after minimum iterations)
        if t ≥ miniterations_before_convergence
            # Check if all states in history are identical
            if length(state_history) == patience
                all_same = true
                # Convert queue to array for easier comparison
                history_array = collect(state_history)
                if length(history_array) > 0
                    first_state = history_array[1]
                    for idx = 2:length(history_array)
                        if any(history_array[idx] .!= first_state)
                            all_same = false
                            break
                        end
                    end
                # Removed extra end statement
                end
                
                if all_same
                    converged = true
                end
            end
        end
        
        t += 1
    end
    
    return frames, energydictionary
end
