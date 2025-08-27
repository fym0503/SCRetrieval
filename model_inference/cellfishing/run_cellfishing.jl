using HDF5
using CellFishing

fid = h5open(string("../../raw_data/Limb-Embryo/sc.h5ad"))
data=read(fid)
counts = zeros(length(data["obs"]["_index"]),length(data["var"]["gene_ids"]))


let counter = 1
  for (index,i) in enumerate(data["X"]["indptr"][2:end]), j in data["X"]["indices"][data["X"]["indptr"][index]+1:i]
    counts[index,j+1] = Int.(floor(data["X"]["data"][counter]))
    counter = counter + 1
  end
end
featurenames = [string(i) for i in 1:length(data["var"]["gene_ids"])]
cellnames = data["obs"]["index"]
counts = Matrix{Int}(counts)

features = CellFishing.selectfeatures(transpose(counts), featurenames)
database = CellFishing.CellIndex(transpose(counts), features, metadata=cellnames)

fid = h5open(string("../../raw_data/Limb-Embryo/st.h5ad"))
data=read(fid)
counts = zeros(length(data["obs"]["_index"]),length(data["var"]["gene_ids"]))
let counter = 1
  for (index,i) in enumerate(data["X"]["indptr"][2:end]), j in data["X"]["indices"][data["X"]["indptr"][index]+1:i]
    counts[index,j+1] =Int.(floor( data["X"]["data"][counter]))
    counter = counter + 1
  end
end
    
featurenames = [string(i) for i in 1:length(data["var"]["gene_ids"])]
cellnames = data["obs"]["index"]
counts = Matrix{Int}(counts)

for k in [100]

  


  

  # Search the database for similar cells; k cells will be returned per query.
  neighbors = CellFishing.findneighbors(k, transpose(counts), featurenames, database)

  # Write the neighboring cells to a file.
  open(string("../../raw_data/Limb-Embryo/cellfishing/st_sc/neighbors_index_",k,".tsv"), "w") do file
      println(file, join(["cell"; string.("n", 1:k)], '\t'))
      for j in 1:length(cellnames)
          print(file, cellnames[j])
          for i in 1:k
              print(file, '\t', database.metadata[neighbors.indexes[i,j]])
          end
          println(file)
      end
  end
  
end
# splits = splits+1

