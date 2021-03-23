from sys import argv
from . import gcd_tools

import numpy as np
from numpy import sqrt,log


# constants
conc_const = 0.358 # 200 * 4/3 * pi * rho_crit in internal units
rho0_const = 0.0286 # 200 /3 * rho_crit in internal units
fourpi = 12.5663706144

# Unit conversions
# Time in Gyr, yr, and seconds
TMUGYR=0.471e0
TMUYR=0.471e9
TMUS=1.488e16
# Mass in Msun and grams
MUSM=1.0e12
MUG=1.99e45
# Dist in kpc, pc and cm
LUKPC=1.0e2
LUPC=1.0e5
LUCM=3.086e23
# Speed in km/s
VUKMS=207.4e0
# Acceleration in dynes/g (cm/s/s)
FUACC = LUCM/(TMUS**2)
# Tide-force in dynes/g/kpc (cm/s/s/kpc)
TIDEU = LUCM/(TMUS**2)/LUKPC
# More constants
G=1.0e0
# min_dt = 1.e5/TMUYR
min_dt = 1.e5/TMUYR



class tideTracker:
    def __init__(self,irunStr,gcdpDir = None):
        if gcdpDir is None:
            gcdpDir = gcd_tools.get_dir(irunStr)
            gcdpDir = gcdpDir.strip()

        iniPramsFile = gcdpDir+"/"+irunStr+"/diskev/ini/input.dat"

        f = open(iniPramsFile,'r')
        #tidesIn = f.readlines()[10:14]
        tidesIn = f.readlines()[12:16]
        f.close()
        tidesIn = [x.split("!")[0] for x in tidesIn]
        tidesIn = [[float(y) for y in x.split()] for x in tidesIn]

        # read into initial conditions
        gp_m = tidesIn[0][0]
        gp_c = tidesIn[0][1]
        gp_t0 = tidesIn[1][0]
        gp_x0 = tidesIn[2][0]
        gp_y0 = tidesIn[2][1]
        gp_z0 = tidesIn[2][2]
        gp_vx0 = tidesIn[3][0]
        gp_vy0 = tidesIn[3][1]
        gp_vz0 = tidesIn[3][2]

        #print(gp_m,gp_c,gp_t0,gp_x0,gp_y0,gp_z0,gp_vx0,gp_vy0,gp_vz0)

        # convert to internal gcdp units
        #global gp_x,gp_y,gp_z,gp_vx,gp_vy,gp_vz,time_sofr,min_dt,gp_a,gp_rho0,gp_nfw_const
        gp_m = gp_m / MUSM

        gp_x0 = gp_x0 / LUPC
        gp_y0 = gp_y0 / LUPC
        gp_z0 = gp_z0 / LUPC

        self.gp_x = gp_x0
        self.gp_y = gp_y0
        self.gp_z = gp_z0

        self.gp_vx = gp_vx0 / VUKMS
        self.gp_vy = gp_vy0 / VUKMS
        self.gp_vz = gp_vz0 / VUKMS

        # calculate constants in internal gcdp units
        self.gp_a = (gp_m/conc_const)**(1./3.) / gp_c
        gp_rho0 = rho0_const * gp_c**3/(log(1+gp_c)-gp_c/(1.+gp_c))
        self.gp_nfw_const = -fourpi * G * gp_rho0 * self.gp_a

        # set up initial time
        self.time_sofar = gp_t0/TMUGYR
        self.big_mass = gp_m*MUSM

    def gp_x_kpc(self):
        return self.gp_x*LUKPC

    def gp_y_kpc(self):
        return self.gp_y*LUKPC

    def gp_z_kpc(self):
        return self.gp_z*LUKPC

    def gp_v_kms(self):
        return np.array([self.gp_vx,self.gp_vy,self.gp_vz])*VUKMS


    # definitions of NFW potential, integration etc
    def nfw_mag(self,d=None):
        if ( d is None ):
            return self.nfw_mag_internal()
#             d=sqrt((self.gp_x**2)+(self.gp_y**2)+(self.gp_z**2))
        normdist = d/self.gp_a/LUKPC
        mag = self.gp_nfw_const * (normdist/(1.+normdist)-log(1.+normdist))/normdist**2
        mag = mag * FUACC
        return mag
    
    
    def nfw_mag_internal(self,d=None):
        if ( d is None ):
            d=sqrt((self.gp_x**2)+(self.gp_y**2)+(self.gp_z**2))
        normdist = d/self.gp_a
        mag = self.gp_nfw_const * (normdist/(1.+normdist)-log(1.+normdist))/normdist**2
        return mag

    def nfw_f_internal(self,x=None,y=None,z=None):
        if ( x is None ):
            x = self.gp_x
        if ( y is None ):
            y = self.gp_y
        if ( z is None ):
            z = self.gp_z
            
        dist = sqrt((x**2)+(y**2)+(z**2))

        fmag = self.nfw_mag_internal(dist)
        fm = fmag

        fmag = fmag/dist

        ax = fmag * (x)
        ay = fmag * (y)
        az = fmag * (z)
        return ax,ay,az,fm

    # position in kpc
    def nfw_rad(self):
        d=sqrt((self.gp_x**2)+(self.gp_y**2)+(self.gp_z**2))
        d=d*LUKPC
        return d
    
    # velocity in km/s
    def nfw_vel(self):
        v=sqrt((self.gp_vx**2)+(self.gp_vy**2)+(self.gp_vz**2))
        v=v*VUKMS
        return v

    def nfw_f(self,x_in,y_in,z_in):
        x = self.gp_x-x_in/LUKPC
        y = self.gp_y-y_in/LUKPC
        z = self.gp_z-z_in/LUKPC
            
        dist = sqrt((x**2)+(y**2)+(z**2))

        fmag = self.nfw_mag_internal(dist)
        fm = fmag

        fmag = fmag/dist*FUACC

        ax = fmag * (x)
        ay = fmag * (y)
        az = fmag * (z)
        return ax,ay,az,fm
    
    def nfw_tide(self,d=None):
        if ( d is None ):
            d=sqrt((self.gp_x**2)+(self.gp_y**2)+(self.gp_z**2))
        normdist = d/self.gp_a
        mag = self.gp_nfw_const * (2*log(1+normdist)/normdist - 2/(1+normdist) - normdist/(1+normdist)**2)/normdist**2
        # mag is in units of force/distance
        return mag*TIDEU

    def update_pos(self,t_in):
        t_in = t_in/TMUYR
        
        #global gp_x,gp_y,gp_z,gp_vx,gp_vy,gp_vz,time_sofar
        if (t_in>=self.time_sofar+min_dt):
            deltat = t_in - self.time_sofar
            dt = min(deltat,min_dt)

            t = self.time_sofar
            while (t<t_in):
                ax,ay,az,fm = self.nfw_f_internal(self.gp_x,self.gp_y,self.gp_z)
                self.gp_x = self.gp_x + dt*self.gp_vx
                self.gp_y = self.gp_y + dt*self.gp_vy
                self.gp_z = self.gp_z + dt*self.gp_vz
                self.gp_vx = self.gp_vx - dt*ax
                self.gp_vy = self.gp_vy - dt*ay
                self.gp_vz = self.gp_vz - dt*az
#                 print(ax,ay,az,dt)
                t = t + dt
            self.time_sofar = t


if __name__ == '__main__':
    irun = argv[1]
    myTides = tideTracker(irun)
    outp = []
    for t in np.arange(0.,5.e9,step=1.e6):
        myTides.update_pos(t)
        outp.append([t,myTides.gp_x,myTides.gp_y,myTides.gp_z,myTides.nfw_rad()])
    outp = np.array(outp)
    np.savetxt("../data/tidetrack{}.dat".format(irun),outp)



