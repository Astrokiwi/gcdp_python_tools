from scipy.io import FortranFile
import numpy as np
import pandas as pd
import pynbody
import os.path
import tidetracker

def gridsize_from_n(n,aspect=1.):
    nx = 1
    ny = 1
    
    if aspect is None:
        aspect = 1.
    
    while nx*ny<n:
        if nx>aspect*ny:
            ny+=1
        else:
            nx+=1
    return nx,ny

solarOFe = 0.914
solarFeH = -2.777

gamma = 5./3.
boltzmann = 1.3806488e-16 # in erg/K
amu = 1.66053892e-24
tempConst = amu/boltzmann

amu_array = pynbody.array.SimArray(amu,"g")                                                                                                                      
tempConst_array = pynbody.array.SimArray(tempConst,"g K erg**-1")                                                                                                                      

bb_integer_keys = { 0:"id",
                 1:"itype"}
bb_float_keys = { 0:"x",
               1:"y",
               2:"z",
               3:"vx",
               4:"vy",
               5:"vz",
               6:"mass",
               7:"rho",
               8:"u"}
met_float_keys = {  0:"ZHe",
                    1:"ZZ",
                    2:"ZC",
                    3:"ZN",
                    4:"ZO",
                    5:"ZNe",
                    6:"ZMg",
                    7:"ZSi",
                    8:"ZFe"}
extra_hydro_float_keys = {  0:"h",
                            1:"divv",
                            2:"alpv",
                            3:"alpu",
                            4:"myu"}

dark_float_keys = {    0:"x",
                       1:"y",
                       2:"z",
                       3:"vx",
                       4:"vy",
                       5:"vz",
                       6:"mass",
                       7:"rho",
                       8:"h"}
dark_integer_keys = {  1:"id"}
unit_conversions = {
                    "x":100., # pos in kpc
                    "y":100.,
                    "z":100.,
                    
                    "vx":207.4, # v in km/s
                    "vy":207.4, 
                    "vz":207.4,
                    
                    "mass":1.e12, # m in Msun
                    
                    "rho":6.77e-26, # density in g/cm^3
                    
                    "u":4.30345382e14, # internal energy in erg/g
                    
                    "h":100. # smoothing/softening in kpc
                    }

tunit = 4.71e8


pynbody_units = {   "x":"kpc",
                    "y":"kpc",
                    "z":"kpc",
                    "vx":"km s**-1",
                    "vy":"km s**-1",
                    "vz":"km s**-1",
                    "mass":"Msol",
                    "rho":"g cm**-3",
                    "u":"erg g**-1",
                    "eps":"kpc",
                    "h":"kpc"
                    }

for z in met_float_keys.values():
    pynbody_units.update({z:"Msol"})

pynbody_conversion_keys = {x:x for x in bb_float_keys.values()}
pynbody_conversion_keys.update({"id":"id"})
pynbody_conversion_keys.update({"eps":"h"})

class PType:
    GAS = 0
    FEED = 1
    STAR = 2
    DARK = 3

class Particle_Stats:
    def __init__(self,time,nprocs,proc_stats):
        self.nb = int(proc_stats["nb"].sum())
        self.ndm = int(proc_stats["ndm"].sum())
        self.ns = int(proc_stats["ns"].sum())
        self.ng = int(proc_stats["ng"].sum())
        self.proc_stats = proc_stats
        self.time = time
        self.nprocs = int(nprocs)

class gcd_data:
    def __init__(self,dir):
        self.set_dir(dir)

    def set_dir(self,dir):
        self.dir = dir
        f = dir+"/diskev/output/ana/ostep.dat"
        self.dump_steps = np.loadtxt(f).astype(np.int)

    def _parse_proc_data(self,f,nb,nvalues,keys,df=None,float_format=True):
        if df is None:
            df = self.particles
        format_string = "f8" if float_format else "i4"
        for ival in range(nvalues):
            indata = np.concatenate([f.read_record('<{}'.format(format_string)) for x in nb])
            if ival in keys:
                key = keys[ival]
                df[key] = indata


    def read_base(self,idump,onlyheader=False):
        if idump<0 or idump>len(self.dump_steps):
            raise ValueError("idump out of range - required 0<={}<={}".format(idump,len(self.dump_steps)))
        self.dump_step = self.dump_steps[idump]
        fname = self.dir+"/diskev/output/data/bbvals%06dn0000"%self.dump_step
#         print(fname)
        f = FortranFile(fname,'r')
        nb,ndm,*x,time = f.read_record('<i4,<i4,<i4,<f8,<f8')[0]
        nprocs = f.read_record('<i4')[0]
        proc_stats = pd.DataFrame(0,index=np.arange(nprocs),columns=("ng","ndm","ns"))
        for i in range(nprocs):
            proc_stats["ng"][i],proc_stats["ndm"][i],proc_stats["ns"][i],*x = f.read_record('4<i4')[0]
        proc_stats["nb"]=proc_stats["ng"]+proc_stats["ns"]

        time*=tunit
        self.dump_stats = Particle_Stats(time,nprocs,proc_stats)
    
        if onlyheader:
            f.close()
            return
    
        self.particles = pd.DataFrame()
        self._parse_proc_data(f,self.dump_stats.proc_stats["nb"],4,bb_integer_keys,float_format=False)
        self._parse_proc_data(f,self.dump_stats.proc_stats["nb"],9,bb_float_keys,float_format=True)
        f.close()
        
        gas_slice = self.particles.itype==0
        star_slice = self.particles.itype>0
        feed_slice = self.particles.itype<0
        self.particles.loc[gas_slice,"itype"] = PType.GAS
        self.particles.loc[star_slice,"itype"] = PType.STAR
        self.particles.loc[feed_slice,"itype"] = PType.FEED
        self.dump_stats.gas_slice = gas_slice
        self.dump_stats.star_slice = star_slice
        self.dump_stats.feed_slice = feed_slice
        self.dump_stats.allstar_slice = (feed_slice) | (star_slice)
        self.dump_stats.dm_slice = np.zeros_like(gas_slice)
        self.dark_loaded = False

    def read_header(self,idump):
        self.read_base(idump,onlyheader=True)



    def read_metals(self):
        fname = self.dir+"/diskev/output/data/bbmets%06dn0000"%self.dump_step
        f = FortranFile(fname,'r')
        for i in range(2):
            f.read_record('<i4')
        
        self._parse_proc_data(f,self.dump_stats.proc_stats["nb"],9,met_float_keys,float_format=True)
        f.close()
        pynbody_conversion_keys.update({x:x for x in met_float_keys.values()})   

    def read_extra_hydro(self):
        fname = self.dir+"/diskev/output/data/bbhyds%06dn0000"%self.dump_step
        f = FortranFile(fname,'r')
        for i in range(2):
            x=f.read_record('<i4')
        
        self._parse_proc_data(f,self.dump_stats.proc_stats["nb"],5,extra_hydro_float_keys,float_format=True)
        f.close()
        pynbody_conversion_keys.update({x:x for x in extra_hydro_float_keys.values()})   


    def read_dark(self):
        fname = self.dir+"/diskev/output/data/bdvals%06dn0000"%self.dump_step
        f = FortranFile(fname,'r')
        for i in range(2+self.dump_stats.nprocs):
            x=f.read_record('<i4')
        
        dm = pd.DataFrame()

        self._parse_proc_data(f,self.dump_stats.proc_stats["ndm"],2,dark_integer_keys,float_format=False,df=dm)
        self._parse_proc_data(f,self.dump_stats.proc_stats["ndm"],9,dark_float_keys,float_format=True,df=dm)
        f.close()
        
        dm['itype'] = np.full((self.dump_stats.ndm),PType.DARK)
        
        self.particles = self.particles.append(dm,sort=False)
        self.dump_stats.dm_slice = self.particles.itype==PType.DARK
        
        # redo slices to account for gas
        self.dump_stats.gas_slice = self.particles.itype==PType.GAS
        self.dump_stats.star_slice = self.particles.itype==PType.STAR
        self.dump_stats.feed_slice = self.particles.itype==PType.FEED
        self.dump_stats.allstar_slice = (self.dump_stats.feed_slice) | (self.dump_stats.star_slice)
        
        self.dark_loaded = True

    
    def convert_units(self):
        for key in unit_conversions:
            if key in self.particles:
                self.particles[key]*=unit_conversions[key]
        
    def to_pynbody(self):
        snap = pynbody.snapshot.new(dm=self.dump_stats.ndm,gas=int(np.sum(self.dump_stats.gas_slice)),star=int(np.sum(self.dump_stats.allstar_slice)))
        gas_fam = lambda x : x.gas 
        star_fam = lambda x : x.star 
        dm_fam = lambda x : x.dm
        if self.dark_loaded:
            fams = (gas_fam,star_fam,dm_fam)
            slices = (self.dump_stats.gas_slice,self.dump_stats.allstar_slice,self.dump_stats.dm_slice)
        else:
            fams = (gas_fam,star_fam)
            slices = (self.dump_stats.gas_slice,self.dump_stats.allstar_slice)
        for fam,slice in zip(fams,slices):
            for pynbody_key,gcd_key in pynbody_conversion_keys.items():
                
                if type(gcd_key) is tuple:
                    unitless_array = np.vstack((self.particles.loc[slice,key] for key in gcd_key)).T
                else:
                    if gcd_key not in self.particles:
                        continue
                    unitless_array = self.particles.loc[slice,gcd_key]
                if pynbody_key in pynbody_units:
                    fam(snap)[pynbody_key] = pynbody.array.SimArray(unitless_array,pynbody_units[pynbody_key])
                else:
                    fam(snap)[pynbody_key] = unitless_array #pynbody.array.SimArray(unitless_array,1)
        return snap

# def find_plane(snap):
#     

def get_dir(run):
    with open("../dirnames.dat") as f:
        for line in f:
            dir_base = line.strip()
            fname = "{}/{}/diskev/output/ana/ostep.dat".format(dir_base,run)
            if os.path.isfile(fname):
                return dir_base
    return None

def extract_times(dir):
    times = []
    
    gr = gcd_data(dir)
    for iline,idump in enumerate(range(len(gr.dump_steps))):
        gr.read_header(idump)
        times.append(gr.dump_stats.time)
    return times

def dump_times(run):
    root_dir = get_dir(run)
    full_dir = "{}/{}".format(root_dir,run)
    times = extract_times(full_dir)
    np.savetxt("../data/timedump{}.dat".format(run),times)

def unglitched_times(run):
    times = np.loadtxt("../data/timedump{}.dat".format(run))
    oldtime = times[-1]
    good_times = [len(times)-1]
    for itime in range(len(times)-1,-1,-1):
        if times[itime]<oldtime:
            good_times.append(itime)
            oldtime=times[itime]
    np.savetxt("../data/timecleaned{}.dat".format(run),good_times[::-1])

def times_to_dumps(time_stops,run):
    fname = "../data/timedump{}.dat".format(run)
    if not os.path.isfile(fname):
        dump_times(run)
    times = np.loadtxt(fname)
    idumps = np.searchsorted(times,time_stops)
    return idumps

def add_tides(snap,td):
    snap["x"]-=td.gp_x*tidetracker.LUKPC
    snap["y"]-=td.gp_y*tidetracker.LUKPC
    snap["z"]-=td.gp_z*tidetracker.LUKPC

    snap["vx"]-=td.gp_vx*tidetracker.VUKMS
    snap["vy"]-=td.gp_vy*tidetracker.VUKMS
    snap["vz"]-=td.gp_vz*tidetracker.VUKMS


# TODO - add these as proper derived arrays
def calc_H(snap):
    if "ZH" not in snap:
        snap.gas["ZH"] = snap.gas["mass"]-(snap.gas["ZHe"]+snap.gas["ZZ"])
        snap.star["ZH"] = snap.star["mass"]-(snap.star["ZHe"]+snap.star["ZZ"])

def calc_met_solar_ratios(snap):
    if "OFe" not in snap:
        snap.gas["OFe"] = np.log10(snap.gas["ZO"]/snap.gas["ZFe"])-solarOFe
        snap.star["OFe"] = np.log10(snap.star["ZO"]/snap.star["ZFe"])-solarOFe
    if "FeH" not in snap:
        calc_H(snap)
        snap.gas["FeH"] = np.log10(snap.gas["ZFe"]/snap.gas["ZH"])-solarFeH
        snap.star["FeH"] = np.log10(snap.star["ZFe"]/snap.star["ZH"])-solarFeH

def calc_nH(snap):
    if "nH" not in snap:
        calc_H(snap)
        snap.gas["nH"] = (snap.gas["ZH"]/snap.gas["mass"] * snap.gas["rho"])/amu_array
        snap.star["nH"] = (snap.star["ZH"]/snap.star["mass"] * snap.star["rho"])/amu_array

def calc_T(snap):
    if "T" not in snap:
        snap.gas["T"] = snap.gas["myu"]*snap.gas["u"]*tempConst_array

def unit_preserving_norm(x):
    return (np.sum(x**2,axis=1))**(1,2)

def unit_preserving_norm2(x):
    return np.sum(x**2,axis=1)


# test package
if __name__ == "__main__":
#     dump_times(1243)
#     for run in ["1800","1801","1804","1805","1806","1807"]:
    for run in ["1808"]:
        dump_times(run)
        unglitched_times(run)
#     x=times_to_dumps([1.e9,2.e9,3.e9],1800)
    
#     import matplotlib.pyplot as plt
#     import matplotlib
# 
#     gr = gcd_data("/srv/djw1g16/gcd/1800")
#     
#     gr.read_base(9)
#     gr.read_extra_hydro()
#     gr.read_dark()
#     gr.convert_units()
#     snap = gr.to_pynbody()

    #pot=pykdgrav.Potential(snap["pos"],snap["mass"],snap["eps"],parallel=True) # slow - run once and save results?
    
#     plt.clf()
#     plt.hist2d(gr.particles.x,gr.particles.y,bins=128,range=((-40.,40.),(-40.,40.)),norm=matplotlib.colors.LogNorm())
#     plt.savefig("../../../figures/xy_test.png",dpi=300)
#     plt.clf()
#     plt.hist2d(gr.particles.x,gr.particles.z,bins=128,range=((-40.,40.),(-40.,40.)),norm=matplotlib.colors.LogNorm())
#     plt.savefig("../../../figures/xz_test.png",dpi=300)
    


