import ROOT

# Open the ROOT file
file = ROOT.TFile.Open("/nfs/dust/cms/user/stadie/2016_94X/roodatasets_newflavor_muon/HDamp_V0.root")

# Access the 'data' object
data = file.Get("data")

# Check if it is indeed a RooDataSet
print(type(data))

# If it’s a RooDataSet, print its entries
if isinstance(data, ROOT.RooDataSet):
    data.Print()  # Prints information about the RooDataSet
    print("Number of entries:", data.numEntries())
else:
    print("The 'data' object is not a RooDataSet.")

#if isinstance(data, ROOT.RooDataSet):
#    for i in range(data.numEntries()):
#        row = data.get(i)
#        row.Print("v")  # Prints the variables for each entry