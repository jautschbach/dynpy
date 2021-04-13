import pandas as pd
import numpy as np
import scipy as sp
import numpy as np
import os
import datetime as dtt
import string
from numpy import linalg as la

def reorder_prnc_comp(nuc,columns=['V11','V22','V33']):
    ordered = sorted(nuc[columns],key=lambda x: abs(float(x)))
    onuc = nuc.copy()
    onuc[columns] = ordered
    return onuc

def extract_efg_adf(system="ADF",traj_dir="./example-data/"):
    print("Analyzing data...")

    trajs = [t.name for t in os.scandir(traj_dir) if t.name.isnumeric()]
    ntraj = len(trajs)
    print("Number of trajectories: "+str(ntraj))

    #get list of subdirectories (iframes)
    exdir = traj_dir+trajs[0]+"/ADF/"
    with os.scandir(exdir) as entries:
        iframes = sorted([entry.name for entry in entries if (entry.is_dir() and entry.name.isdigit())],key=int)
    #iframes = [str(i).zfill(4) for i in range(7000,20001)]
    nframes = len(iframes)

    #get number of atoms per frame
    inp = exdir+iframes[0]+'/'+iframes[0]+'-scf.inp'
    with open(inp, 'r') as f:
        lines = f.readlines()
    atoms = False
    nat = 0
    for line in lines:
        if "end" in line or "END" in line:
            atoms = False
        if atoms:
            if line.strip():
                #print(line)
                nat += 1
        if "atoms" in line or "ATOMS" in line:
            atoms = True

    print("Number of frames: "+str(nframes))
    print("Number of atoms per frame: "+str(nat))

    missing = np.empty((int(ntraj*nat*(1.05)*nframes), ), dtype = [('traj','O'), ('frame','O')])
    missingfile = './missing-ADF.csv'
    m = 0

    if os.path.isfile(missingfile):
        os.remove(missingfile)

    #create array data structure
    #print("creating array data structure...")
    data = np.empty((int(ntraj*nat*(1.05)*nframes), ), dtype = [('system', 'O'), ('traj','i8'), ('frame','i8'),
                                        ('time','f8'), ('label','i8'), ('symbol','O'),
                                          ('Vxx', 'f8'), ('Vxy', 'f8'), ('Vxz', 'f8'),
                                          ('Vyx', 'f8'), ('Vyy', 'f8'), ('Vyz', 'f8'),
                                          ('Vzx', 'f8'), ('Vzy', 'f8'), ('Vzz','f8'),
                                          ('V11', 'f8'), ('V22', 'f8'), ('V33', 'f8'),
                                          ('Eta', 'f8'), ('ComputeTime', 'f8')])

    data.fill(np.NaN)
    #print(data)

    idx=-1
    for itr, traj in enumerate(trajs):
        #define directory to extract data from. Must contain subdirectories 0000..XXXX that corresond to frames
        data_dir = traj_dir+traj+"/ADF/"

        #get list of subdirectories (iframes)
        with os.scandir(data_dir) as entries:
            iframes = sorted([entry.name for entry in entries if (entry.is_dir() and entry.name.isdigit())],key=int)

        catch1 = "EFG and ESR Q-TENSOR"
        catch2 = "Electron Density at Nuclei"
        print("Trajectory "+traj+": looping through frame directories...")
        for i, iframe in enumerate(iframes):
            target_dir = data_dir + iframe+'/'
            output = target_dir + iframe + "-scf.out"
            #print(output)
            err = target_dir + iframe + ".err"
            #print(output)
            #print(err)
            lines = None
            ti = None
            cancelled = False

            if os.path.isfile(err):
                with open(err) as f:
                    try:
                        errlines = f.readlines()
                        errlines = "".join(errlines)
                        #print(errlines)
                    except UnicodeDecodeError:
                        print("Unknown Error occurred for " + str(traj) + '-' + iframe)
                        missing[m] = (traj,iframe)
                        m+=1
                        continue
                #print(errlines)
                if "TIME LIMIT" in errlines:
                    print(str(traj) + '-' + iframe + " cancelled due to time limit")
                    missing[m] = (traj,iframe)
                    m+=1
                    continue
                if "PREEMPTION" in errlines:
                    print(str(traj) + '-' + iframe + " cancelled due to preemption")
                    missing[m] = (traj,iframe)
                    m+=1
                    continue
                if "ERROR DETECTED\n" in errlines:
                    print(str(traj) + '-' + iframe + " errored")
                    missing[m] = (traj,iframe)
                    m+=1
                    continue
                if "CANCELLED" in errlines:
                    print(str(traj) + '-' + iframe + " cancelled")
                    cancelled = True
                    #missing[m] = (traj,iframe)
                    #m+=1
                    #continue

            if os.path.isfile(output):
                #print("reading output file...")
                with open(output) as f:
                    lines = f.readlines()
                if not lines:
                    print("No output for " + str(traj) + '-' +  iframe)
                    missing[m] = (traj,iframe)
                    m+=1
                    continue
            else:
                print("No output for " + str(traj) + '-' + iframe)
                missing[m] = (traj,iframe)
                m+=1
                continue

            inblock = False
            end = lines[-1]
            end = end.replace('<','').replace('>','').replace('-',':').strip().strip(string.ascii_letters).strip().replace(' ',':').split(':')
            try:
                tf = dtt.datetime(int(end[1]),1,int(end[0]),int(end[2]),int(end[3]),int(end[4]))
            except (IndexError,ValueError):
                print("No timing data. " + str(traj) + '-' + iframe + " likely didn't run")
                missing[m] = (traj,iframe)
                m+=1
                continue


            for j, line in enumerate(lines):

                if "NOT CONVERGED" in line or "MODERATELY CONVERGED" in line:
                    print(str(traj) + "-" + iframe + " not converged")
                    print(j, line)
                    #with open(missing, 'a') as f:
                    #    f.write(str(frame)+" ")
                    break

                if "RunTime" in line and not ti:
                    start = ":".join(line.split()[3:5])
                    #print(start)
                    start = start.replace('-',':').strip(string.ascii_letters).split(':')
                    try:
                        ti = dtt.datetime(int(start[1]),1,int(start[0]),int(start[2]),int(start[3]),int(start[4]))
                        compute_time = dtt.timedelta.total_seconds(tf-ti)
                    except ValueError:
                        print("Timing issues for " + str(traj) + '-' + iframe)
                        missing[m] = (traj,iframe)
                        m+=1
                        continue

                if "COMMENT" in line:
                    frame = int(lines[j+1].split()[1].strip(','))
                    timeps = float(lines[j+1].split()[-1])

                if catch1 in line:
                    if cancelled:
                        print("EFG data found")

                    inblock = True
                if catch2 in line:
                    inblock = False

                if inblock:
                    if "EFG-tensor" in line:
                        label = lines[j-2].split()[1]
                        symbol = lines[j].split()[-1]
                        vxx = lines[j+2].split()[1]
                        vxy = lines[j+2].split()[2]
                        vxz = lines[j+2].split()[3]
                        vyx = lines[j+3].split()[1]
                        vyy = lines[j+3].split()[2]
                        vyz = lines[j+3].split()[3]
                        vzx = lines[j+4].split()[1]
                        vzy = lines[j+4].split()[2]
                        vzz = lines[j+4].split()[3]
                        v11 = lines[j+13].split()[0]
                        v22 = lines[j+13].split()[1]
                        v33 = lines[j+13].split()[2]
                        eta = lines[j+19].split()[13]
                        idx += 1
                        #print((itr+1)*i*nat+int(label)-1)
                        #datar = np.array([tuple([solvent, traj, frame, timeps, label, symbol, vxx, vxy, vxz, vyx, vyy, vyz, vzx, vzy, vzz, v11, v22, v33, q11, q22, q33, eta, compute_time])], dtype = data.dtype)
                        #print(datar)
                        #print(datar.shape)
                        #print(data.shape)
                        #data = np.concatenate([data, datar], axis=0)
                        data[idx] = (system, traj, frame, timeps, label, symbol, vxx, vxy, vxz, vyx, vyy, vyz, vzx, vzy, vzz, v11, v22, v33, eta, compute_time)
                #if j==(len(lines)-1):

    df = pd.DataFrame(data)
    df.dropna(inplace = True)
    dfr = df.apply(reorder_prnc_comp,axis=1)

    dfr.to_csv(system+'-efg.csv', index = False)
    #df.to_hdf(target_dir+'EFG_'+solvent+'.hdf', 'df')
    print("EFG data written to "+ system +"-efg.csv")
    if any(missing):
        dfm = pd.DataFrame(missing)
        dfm.dropna(inplace = True)
        dfm.to_csv(missingfile, index = False)
        print("missing calcs written to " + missingfile)

    print("Done")
    return(dfr)


def extract_efg_qe(system="GIPAW",traj_dir="./example-data/"):
    print("Analyzing data...")

    trajs = [t.name.zfill(2) for t in os.scandir(traj_dir) if t.name.isnumeric()]
    ntraj = len(trajs)
    print("Number of trajectories: "+str(ntraj))

    #get list of subdirectories (iframes)
    exdir = traj_dir+trajs[0]+"/GIPAW/"
    with os.scandir(exdir) as entries:
        iframes = sorted([entry.name for entry in entries if (entry.is_dir() and entry.name.isdigit())],key=int)
    nframes = len(iframes)

    #get number of atoms per frame
    inp = exdir+iframes[0]+'/'+iframes[0]+'-scf.inp'
    with open(inp, 'r') as f:
        lines = f.readlines()
    atoms = False
    nat = 0
    symbols = []
    for line in lines:
        if "K_POINTS automatic" in line:
            atoms = False
        if atoms:
            if line.strip():
                #print(line)
                nat += 1
                symbols.append(line[0])
        if "ATOMIC_POSITIONS" in line:
            atoms = True

    print("Number of frames: "+str(nframes))
    print("Number of atoms per frame: "+str(nat))
    #print(symbols)

    missing = np.empty((int(ntraj*nat*(1.05)*nframes), ), dtype = [('traj','O'), ('frame','O')])
    missingfile = './missing-GIPAW.csv'
    if os.path.isfile(missingfile):
        os.remove(missingfile)
    m = 0

    time = []
    frames = []
    nl = []
    dummy=[[np.nan]*3]*3*nat

    for traj in trajs:
        data_dir = traj_dir + traj+"/GIPAW/"
        with os.scandir(data_dir) as entries:
            iframes = sorted([entry.name for entry in entries if (entry.is_dir() and entry.name.isdigit())],key=int)

        print("Trajectory "+traj+": looping through frame directories...")
        for i, iframe in enumerate(iframes):
            efgout = data_dir+iframe + "/" + iframe + "-efg.out"
            inp = data_dir+iframe + "/" + iframe + "-scf.inp"
            scfout= data_dir+iframe + "/" + iframe + "-scf.out"
            lines = None
            catch1 = False

            with open(inp, 'r') as g:
                ilines = g.readlines()

            for line in ilines:
                if "!frame:" in line:
                    frames.append(line.split()[1].strip(','))
                    time.append(line.split()[-1])
                    break

            if os.path.isfile(efgout):
                with open(efgout, 'r') as f:
                    lines = f.readlines()
            else:
                print("No output file for " +str(traj)+"-"+iframe)
                missing[m] = (traj,iframe)
                m+=1
                for dum in dummy:
                    nl.append(dum)
                continue

            for j, line in enumerate(lines):
                if catch1:
                    if "NQR/NMR SPECTROSCOPIC PARAMETERS" in line:
                        break
                    elif any(sym in line for sym in list(set(symbols))):
                        ll = line.strip().split()
                        for n,l in enumerate(ll):
                            try:
                                ll[n]=float(l)
                            except:
                                pass
                        nl.append(ll[2:])
                if "----- total EFG (symmetrized) -----" in line:
                    catch1=True

            if catch1==False:
                print("No data found in output for " +str(traj)+"-"+iframe)
                missing[m] = (traj,iframe)
                m+=1
                for dum in dummy:
                    nl.append(dum)
                continue

            with open(scfout,'r') as f:
                outlines = f.readlines()
            #print(outlines)
            if "     convergence NOT achieved after 200 iterations: stopping\n" in outlines:
                print(str(traj)+"-"+iframe + " not converged!")
                missing[m] = (traj,iframe)
                m+=1
                for dum in dummy:
                    nl.append(dum)
                continue

    #print(nl)
    df=pd.DataFrame(nl)
    #print(df[4220:4225])
    #print("selected frames= "+str(sframes))
    data = np.empty((ntraj*nframes*nat, ), dtype = [('system', 'O'),('traj','i8'),('frame', 'i8'), ('time','f8'),
                                         ('label', 'i8'), ('symbol','O'),('Vxx', 'f8'),
                                         ('Vxy', 'f8'), ('Vxz', 'f8'), ('Vyx', 'f8'),
                                         ('Vyy', 'f8'), ('Vyz', 'f8'), ('Vzx', 'f8'),
                                         ('Vzy', 'f8'), ('Vzz','f8'),('V11', 'f8'),
                                         ('V22','f8'),('V33', 'f8'), ('eta','f8')])
    q=0
    for traj in range(ntraj):
        for i in range(nframes):
            for N in range(nat):
                x=df.loc[3*(traj*i*nat)+3*(i*nat)+(3*N)]
                y=df.loc[3*(traj*i*nat)+3*(i*nat)+(3*N)+1]
                z=df.loc[3*(traj*i*nat)+3*(i*nat)+(3*N)+2]
                #print(x)
                #print(y)
                #print(z)

                vxx=x[0]
                vxy=x[1]
                vxz=x[2]
                vyx=y[0]
                vyy=y[1]
                vyz=y[2]
                vzx=z[0]
                vzy=z[1]
                vzz=z[2]

                tens = [[0,0,0],[0,0,0],[0,0,0]]
                tens[0] = [vxx, vxy, vxz]
                tens[1] = [vyx, vyy, vyz]
                tens[2] = [vzx, vzy, vzz]

                try:
                    eig, vec = la.eig(tens)
                    #print(eig)
                    #print(vec)
                    v = sorted(eig, key=lambda x: np.abs(x))
                    v11 = v[0]
                    v22 = v[1]
                    v33 = v[2]
                    eta = (v11-v22)/v33
                except np.linalg.linalg.LinAlgError:
                    v11 = np.NAN
                    v22 = np.NAN
                    v33 = np.NAN
                    eta = np.NAN
                data[q] = (system,traj+1,frames[i],time[i],N,symbols[q%len(symbols)], vxx, vxy, vxz, vyx, vyy, vyz, vzx, vzy, vzz,v11,v22,v33,eta)
                q+=1
    df=pd.DataFrame(data)
    df.to_csv(system+"-efg.csv",index=False)
    print("EFG data written to "+ system +"-efg.csv")
    if any(missing):
        dfm = pd.DataFrame(missing)
        dfm.dropna(inplace = True)
        dfm.to_csv(missingfile, index = False)
        print("missing calcs written to " + missingfile)

    print("Done")
    return(df)
