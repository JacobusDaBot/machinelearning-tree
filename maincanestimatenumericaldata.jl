using CSV, DataFrames,Random
using Statistics
import Random

mutable struct node
    column::Union{String, Nothing}
    fnct::Union{Function, Nothing}
    left::Union{node, Nothing}
    right::Union{node, Nothing}
    prediction::Union{Number, Nothing}  # For leaf nodes

    # Constructor for split nodes
    function node(column::String, funct::Function)
        new(column, funct, nothing, nothing, nothing)
    end
    
    # Constructor for leaf nodes with prediction
    function node(prediction::Number)
        new(nothing, nothing, nothing, nothing, prediction)
    end
end

# Simplified display function without recursion
function Base.show(io::IO, n::node)
    if n.prediction !== nothing
        print(io, "Leaf(prediction=$(n.prediction))")
    else
        print(io, "Split(column=$(n.column))")
    end
end

function Base.show(io::IO, n::node, depth::Int)
    indent = repeat("  ", depth)
    if n.prediction !== nothing
        println(io, indent * "Leaf(prediction=$(n.prediction))")
    else
        if n.left !== nothing
            show(io, n.left, depth + 1)
        end
                println(io, indent * "Split(column=$(n.column))")
        if n.right !== nothing
            show(io, n.right, depth + 1)
        end
    end
end



function entropy(X,estimatingfield)
    nr = nrow(X)
    if nr == 0
        return 0.0
    end
    #println(estimatingfield)
    return  var(X[!,estimatingfield])
end
function gain(S,A,B,estimatingfield)
    eS=entropy(S,estimatingfield)

    nr=nrow(S)
    weB = entropy(B,estimatingfield) * (nrow(B)/nr)
    weA = entropy(A,estimatingfield) * (nrow(A)/nr)
    return abs(eS- weA-weB )
    #High Gain = Good split (much purer children)
    #Low Gain = Poor split (children still impure)
end
function variance(X, estimatingfield)
    nr = nrow(X)
    if nr <= 1
        return 0.0
    end
    return var(X[!, estimatingfield])
end

function regression_gain(S, A, B, estimatingfield)
    # For regression, we use variance reduction (similar to information gain)
    # This is also called "variance impurity"
    eS = variance(S, estimatingfield)
    nr = nrow(S)
    
    if nr == 0
        return 0.0
    end
    
    weA = variance(A, estimatingfield) * (nrow(A) / nr)
    weB = variance(B, estimatingfield) * (nrow(B) / nr)
    
    # Variance reduction (always positive when split improves purity)
    gain_val = eS - weA - weB
    
    # Gain should be non-negative for valid splits
    return max(0.0, gain_val)
end

function getGoodGain(complete_df, colname, type::String, estimatingfield)
    if type == "n"
        sorted_vals = sort(unique(complete_df[!, colname]))
        best_gain = 0.0
        best_point = sorted_vals[1]  # Default to first value
        
        for i in 1:(length(sorted_vals)-1)
            split_point = (sorted_vals[i] + sorted_vals[i+1]) / 2
            left_mask = complete_df[!, colname] .<= split_point
            right_mask = .!left_mask
            
            left = complete_df[left_mask, :]
            right = complete_df[right_mask, :]
            
            if nrow(left) > 0 && nrow(right) > 0
                current_gain = regression_gain(complete_df, left, right, estimatingfield)
                if current_gain > best_gain
                    best_gain = current_gain
                    best_point = split_point
                end
            end
        end
        
        println("Best gain for numerical $colname: $best_gain at $best_point")
        return best_gain, (x) -> x .<= best_point
        
    elseif type == "b"
        unique_vals = unique(complete_df[!, colname])
        if length(unique_vals) != 2
            return 0.0, (x) -> falses(length(x))
        end
        
        split_val = unique_vals[1]
        left_mask = complete_df[!, colname] .== split_val
        right_mask = .!left_mask
        
        left = complete_df[left_mask, :]
        right = complete_df[right_mask, :]
        
        current_gain = regression_gain(complete_df, left, right, estimatingfield)
        println("Gain for binary $colname: $current_gain")
        return current_gain, (x) -> x .== split_val
        
    elseif type == "c"
        unique_cats = unique(complete_df[!, colname])
        
        if length(unique_cats) <= 1
            return 0.0, (x) -> falses(length(x))
        end
        
        best_gain = 0.0
        best_split = nothing
        
        # For regression, sort by mean of target variable instead of survival rate
        cat_stats = []
        for cat in unique_cats
            subset = complete_df[complete_df[!, colname] .== cat, :]
            if nrow(subset) > 0
                mean_value = mean(subset[!, estimatingfield])
                push!(cat_stats, (category=cat, mean_val=mean_value, size=nrow(subset)))
            end
        end
        
        # Sort categories by mean of target variable
        sort!(cat_stats, by=x -> x.mean_val)
        
        # Try different binary splits
        for k in 1:(length(cat_stats)-1)
            left_cats = [cat_stats[i].category for i in 1:k]
            left_mask = [val in left_cats for val in complete_df[!, colname]]
            right_mask = .!left_mask
            
            left = complete_df[left_mask, :]
            right = complete_df[right_mask, :]
            
            if nrow(left) > 0 && nrow(right) > 0
                current_gain = regression_gain(complete_df, left, right, estimatingfield)
                if current_gain > best_gain
                    best_gain = current_gain
                    best_split = left_cats
                end
            end
        end
        
        println("Best gain for categorical $colname: $best_gain")
        if best_split !== nothing
            return best_gain, (x) -> [val in best_split for val in x]
        else
            return 0.0, (x) -> falses(length(x))
        end
    end
end
function pickBestSetForNode(complete_df, column_types, estimatingfield)
    best_gain = 0.0
    best_funct = nothing
    colbest = nothing
    
    for (colname, curtype) in column_types
        if colname == estimatingfield
            continue
        end
        
        current_gain, funct = getGoodGain(complete_df, colname, curtype, estimatingfield)
        println("$colname ($curtype): gain = $current_gain")
        
        if current_gain > best_gain
            best_funct = funct
            best_gain = current_gain
            colbest = colname
        end
    end
    
    if colbest === nothing
        println("No good split found")
    else
        println("Best column: $colbest with gain: $best_gain")
    end
    
    return colbest, best_funct
end

function maketree(complete_df,column_types,estimatingfield,avgparent=0)
        # Base case: if no more features or dataset is empty

    if isempty(column_types) || nrow(complete_df) == 0
        if nrow(complete_df) == 0
            return node(-5)
        end
        return  node(mean(complete_df[!,estimatingfield]))
    end
    println(complete_df[!,estimatingfield])
    average=mean(complete_df[!,estimatingfield])
    colbest, fnct = pickBestSetForNode(complete_df, column_types,estimatingfield)
    
    # If no good split found, create leaf node
    if colbest === nothing ||colbest ==="___"
        if nrow(complete_df) == 0
            return node(-5)
        end
        return  node(average)

    end
    
    root = node(colbest, fnct)
    remaining_types = copy(column_types)
    delete!(remaining_types, colbest)
    
    # Create boolean masks
    mask_left = fnct(complete_df[!, colbest])
    mask_right = .!mask_left
    #println("mask_left: $(sum(mask_left))")
    #println("mask_right: $(sum(mask_right))")
    # Build subtrees
    if sum(mask_left) > 0
        root.left = maketree(complete_df[mask_left, :], remaining_types,estimatingfield,average)
    else
        if nrow(complete_df) == 0
            return node(-5)
        end
        return  node(average)
    end
    
    if sum(mask_right) > 0
        root.right = maketree(complete_df[mask_right, :], remaining_types,estimatingfield,average)
    else
        if nrow(complete_df) == 0
            return node(-5)
        end
        return  node(average)
    end

    return root
end
function predict(tree::node, row::DataFrameRow)
    if tree.prediction !== nothing
        # Leaf node - return prediction
        return tree.prediction
    else
        # Split node - follow the appropriate branch
        column_value = row[Symbol(tree.column)]
        if tree.fnct([column_value])[1]  # Apply split function
            return predict(tree.left, row)
        else
            return predict(tree.right, row)
        end
    end
end

function main()

     df_select = CSV.read("titanic.csv",header=true, DataFrame; select=[2,3,5,6,7,8,10,11,12]) 
    complete_df = dropmissing(df_select)
    i=1
    for x in complete_df[!, "Cabin"] 
        complete_df[i, "Cabin"]  = string(x[1])
        i+=1
    end
    subset_size = 0.5
    random_indices = randsubseq(1:nrow(complete_df), subset_size)
    random_subset = complete_df[random_indices, :]
    estimatingfield="Age"
    column_types = Dict(
        "Survived"=>"b",
        "Pclass" => "c",    # Categorical ordinal
        "Sex" => "b",       # Binary
        "Age" => "n",       # Numerical
        "SibSp" => "c",     # Numerical
        "Parch" => "c",     # Numerical
        "Fare" => "n",       # Numerical
        "Cabin"=> "c",
        "Embarked"=> "c"
    )
    root=maketree(random_subset,column_types,estimatingfield,mean(complete_df[!,estimatingfield]))
    println("\nDecision Tree:")
    show(stdout, root, 0)
    println()
    testsubset=first(complete_df, 30)
    
    avgerror=0
    for x in eachrow(testsubset)
        
        abserror=abs(x[estimatingfield]-predict(root,x))
        println("$(x[estimatingfield])-$(predict(root,x))=$abserror ")
        avgerror+=abserror<=15
    end
    
    println(sum(complete_df[!,estimatingfield])/nrow(complete_df))
    println(avgerror/30)
end

main()






