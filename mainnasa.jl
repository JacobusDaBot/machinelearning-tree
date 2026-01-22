using CSV, DataFrames,Random

import Random

mutable struct node
    column::Union{String, Nothing}
    fnct::Union{Function, Nothing}
    left::Union{node, Nothing}
    right::Union{node, Nothing}
    prediction::Union{Bool, Nothing}  # For leaf nodes

    # Constructor for split nodes
    function node(column::String, funct::Function)
        new(column, funct, nothing, nothing, nothing)
    end
    
    # Constructor for leaf nodes with prediction
    function node(prediction::Bool)
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



function entropy(X)
    nr = nrow(X)
    if nr == 0
        return 0.0
    end
    
    b = count(X.PHA .== 1)  # PHA
    a = count(X.PHA .== 0)  # Died
    
    entropy_val = 0.0
    if a > 0
        p_a = a/nr
        entropy_val -= p_a * log2(p_a)
    end
    if b > 0
        p_b = b/nr
        entropy_val -= p_b * log2(p_b)
    end
    return entropy_val
end
function gain(S,A,B)
    eS=entropy(S)

    nr=nrow(S)
    weB = entropy(B) * (nrow(B)/nr)
    weA = entropy(A) * (nrow(A)/nr)
    return eS- weA-weB 
    #High Gain = Good split (much purer children)
    #Low Gain = Poor split (children still impure)
end
function getGoodGain(complete_df,colname,type::String)
    if type=="n"
        sorted_vals = sort(unique(complete_df[!, colname]))
        best_point = 0.0
        goodgain=0
        #println((sorted_vals))
        for i in 1:(length(sorted_vals)-1)

            split_point = (sorted_vals[i] + sorted_vals[i+1]) / 2
            A    = complete_df[!, colname] .<= split_point
            B    =.!A
            left = complete_df[A, :]
            right = complete_df[B, :]
            #println(nrow(left) +nrow(right) )
            if nrow(left) > 0 && nrow(right) > 0
                gn1=gain(complete_df,left,right)
                #println(gn1)
                if gn1>goodgain
                    goodgain=gn1
                    best_point = split_point
                end
            end
        end
        #println("Best gain for numerical $colname: $goodgain at $best_point")
        return goodgain, (x) -> x .<= best_point
    end
    if  type =="b"
        unique_vals = unique(complete_df[!, colname])
        if length(unique_vals) != 2
            return 0.0, (x) -> falses(length(x))
        end
        # Use the first unique value as the split
        split_val = unique_vals[1]
        A = complete_df[!, colname] .== split_val
        B = .!A
        left = complete_df[A, :]
        right = complete_df[B, :]
        current_gain = gain(complete_df, left, right)
        #println("Gain for binary $colname: $current_gain")
        return current_gain, (x) -> x .== split_val
    end
    if type=="c"
        unique_cats = unique(complete_df[!, colname])
    
        if length(unique_cats) <= 1
            return 0.0, nothing
        end
        
        best_gain = -Inf
        best_split = nothing
        
        # For categorical variables, we need to find the best binary split
        # One common approach: try all possible binary partitions
        # For efficiency, we'll use a simpler approach: sort by target proportion
        
        # Calculate survival rate for each category
        cat_stats = []
        for cat in unique_cats
            subset = complete_df[complete_df[!, colname] .== cat, :]
            survival_rate = nrow(subset) > 0 ? count(subset.PHA .== 1) / nrow(subset) : 0.0
            push!(cat_stats, (category=cat, rate=survival_rate, size=nrow(subset)))
        end
        
        # Sort categories by survival rate
        sort!(cat_stats, by=x -> x.rate)
        
        # Try different splits (group first k categories vs rest)
        for k in 1:(length(cat_stats)-1)
            left_cats = [cat_stats[i].category for i in 1:k]
            right_cats = [cat_stats[i].category for i in (k+1):length(cat_stats)]
            
            left_mask = [val in left_cats for val in complete_df[!, colname]]
            right_mask = [val in right_cats for val in complete_df[!, colname]]
            
            left = complete_df[left_mask, :]
            right = complete_df[right_mask, :]
            
            if nrow(left) > 0 && nrow(right) > 0
                gain_val = gain(complete_df, left, right)
                
                if gain_val > best_gain
                    best_gain = gain_val
                    best_split = (left_cats, right_cats)
                end
            end
        end
        
        #println("Best gain for categorical $colname: $best_gain")
        if best_split !== nothing
            return best_gain, (x) -> [val in best_split for val in x]
        else
            return 0.0, (x) -> falses(length(x))
        end
    end
end
function pickBestSetForNode(complete_df,column_types)
    bestgain=0
    bestfunct=nothing
    colbest="PHA"
    for (colname,curtype) in (column_types)

        complete_df[((complete_df[!, colname]) .== 1), :]
        if colname==="PHA"
            continue
        end
        #println(colname*" "*curtype)
        gn,funct=getGoodGain(complete_df,colname,curtype)
        if gn>bestgain
            bestfunct=funct
            bestgain=gn
            colbest=colname
        end
    end
    #println(colbest)
    #println(bestgain)
    #println("$colbest : $bestgain")
    return colbest,bestfunct
end

function maketree(complete_df,column_types)
        # Base case: if no more features or dataset is empty
    if isempty(column_types) || nrow(complete_df) == 0
        survived_count = count(complete_df.PHA .== 1)
        died_count = count(complete_df.PHA .== 0)
        leaf_value = survived_count >= died_count
        return node(leaf_value)
    end
    
    colbest, fnct = pickBestSetForNode(complete_df, column_types)
    
    # If no good split found, create leaf node
    if colbest === nothing ||colbest ==="PHA"
        survived_count = count(complete_df.PHA .== 1)
        died_count = count(complete_df.PHA .== 0)
        leaf_value = survived_count >= died_count
        return node(leaf_value)
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
        root.left = maketree(complete_df[mask_left, :], remaining_types)
    else
        # Empty branch - create leaf with parent majority
        survived_count = count(complete_df.PHA .== 1)
        died_count = count(complete_df.PHA .== 0)
        leaf_value = survived_count >= died_count
        root.left = node(leaf_value)
    end
    
    if sum(mask_right) > 0
        root.right = maketree(complete_df[mask_right, :], remaining_types)
    else
        # Empty branch - create leaf with parent majority
        survived_count = count(complete_df.PHA .== 1)
        died_count = count(complete_df.PHA .== 0)
        leaf_value = survived_count >= died_count
        root.right = node(leaf_value)
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


    df_select = CSV.read("WISE_NEACOMET_DISCOVERY_STATISTICS_rows.csv",header=true, DataFrame; select=[3,4,5,6,7,8,9,10]) 
    complete_df = dropmissing(df_select)
    complete_df[!, "PHA"] = complete_df[!, "PHA"] .== "Y"
    i=1
    println(nrow(complete_df))
    subset_size = 0.5
    random_indices = randsubseq(1:nrow(complete_df), subset_size)
    random_subset = complete_df[random_indices, :]
    println(nrow(random_subset))
    column_types = Dict(
        "H (mag)"=>"n",
        "MOID (AU)"=>"n",
        "q (AU)"=>"n",
        "Q (AU)"=>"n",
        "period (yr)"=>"n",
        "i (deg)"=>"n",
        "PHA"=>"b",
        "Orbit Class"=>"c",
    )
    root=maketree(random_subset,column_types)
    println("\nDecision Tree:")
    show(stdout, root, 0)
    println()
    testsubset= complete_df[((complete_df[!, "PHA"]) .== 1), :]
    testsubset=vcat(first(complete_df, 30),testsubset)
    actualsurvived=0
    actualdied=0
    correct=0
    incorrect=0
    for x in eachrow(testsubset)
        #println(x)
        actualsurvived+=x["PHA"]==true
        actualdied+=x["PHA"]==false
        correct+=x["PHA"]==predict(root,x)
        incorrect+=x["PHA"]!=predict(root,x)
    end
    println(actualsurvived)
    println(actualdied)
    ic=(correct/(nrow(testsubset)))
    iw=(incorrect/(nrow(testsubset)))
    println(ic)
    println(iw)
    println(ic+iw)
    #println(prediction)
end

main()






