from sklearn.externals import joblib #SOSO
from rdkit.Chem.Draw import IPythonConsole #Needed to show molecules+
from rdkit.Chem import Draw
from rdkit.Chem import AllChem as Chem
import gzip
import os
import sys
sys.path.append("/home/samy/Bureau/mazigh/PSC/Code/SYBA/scscore") #indique le chemin pour le scscore
sys.path.append("/home/samy/Bureau/mazigh/PSC/Code/SYBA/SAscore/src")
#sys.setdefaultencoding('utf-8')
Type = sys.getfilesystemencoding()
from syba.syba import SybaClassifier, SmiMolSupplier
from scscore.standalone_model_numpy import SCScorer
import sascorer as sa
from sklearn.ensemble import RandomForestClassifier

# needed to calculate complexities
from nonpher import complex_lib as cmplx

from datetime import datetime
print(datetime.now(), " 1")#1
nBits = 1024
if not os.path.exists("../data/rf.pkl"):
    print(datetime.now(), " 2")#2
    syn_fps = [Chem.GetMorganFingerprintAsBitVect(spls[0],2,nBits=nBits) for spls in SmiMolSupplier(gzip.open("../data/structures_2.csv.gz", mode="rt"), header=True, smi_col=1)]
    syn_classes = [1 for x in range(len(syn_fps))]
    print(datetime.now(), " 3")#3
    non_fps = [Chem.GetMorganFingerprintAsBitVect(spls[0],2,nBits=nBits) for spls in SmiMolSupplier(gzip.open("../data/structures_1.csv.gz", mode="rt"), header=True, smi_col=2)]
    non_classes = [0 for x in range(len(non_fps))]
    print(datetime.now(), " 4")#4
    fps = syn_fps + non_fps
    classes = syn_classes + non_classes

    clf = RandomForestClassifier(n_estimators=5) #orginaly at 100
    print(datetime.now(), " 5")#5
    clf.fit(fps, classes)
    print(datetime.now(), " 6")#6
    joblib.dump(clf, "../data/rf.pkl")
    print(datetime.now(), " 7")#7
else:
    clf = joblib.load("../data/rf.pkl")
    
print(datetime.now(), " 8")
fp = Chem.GetMorganFingerprintAsBitVect(mol,2,nBits=nBits)
print(clf.predict([fp])[0], clf.predict_proba([fp])[0])
print(datetime.now(), " 9")
