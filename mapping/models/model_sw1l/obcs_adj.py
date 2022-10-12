import numpy as np 

###########################################################################
#                  Boundary conditions functions                          #
###########################################################################


def init_bc_adj(M,adu,adv,adh,He,hbcx,hbcy,t0=0):
    
    # Init 
    adHe2d_incr = np.zeros(M.X.shape)
    adhbcx1d_incr = np.zeros([M.omegas.size,2,2,M.bc_theta.size,M.nx])
    adhbcy1d_incr = np.zeros([M.omegas.size,2,2,M.bc_theta.size,M.ny])
    
    aduS,advS,adhS,aduN,advN,adhN,aduW,advW,adhW,aduE,advE,adhE = \
        update_borders_adj(M,adu,adv,adh)
    
    adHeS = np.zeros(M.nx)
    adHeN = np.zeros(M.nx)
    adHeW = np.zeros(M.ny)
    adHeE = np.zeros(M.ny)
    
    for j,w in enumerate(M.omegas):
        for i,theta in enumerate(M.bc_theta):
            ###############################################################
            # South
            ###############################################################
            fS = (M.f[0,:]+M.f[1,:])/2
            HeS = (He[0,:]+He[1,:])/2
            k = np.sqrt((w**2-fS**2)/(M.g*HeS))
            kx = np.sin(theta) * k
            ky = np.cos(theta) * k
            kxy_h = kx*M.X[0,:] + ky*M.Y[0,:]
            kxy_v = kx*M.Xv[0,:] + ky*M.Yv[0,:]
            phase_h =  w*t0 - kxy_h
            phase_v =  w*t0 - kxy_v
            
            
            adkx = np.zeros(M.nx)
            adky = np.zeros(M.nx)
            adkxy_v = np.zeros(M.nx)
            adkxy_h = np.zeros(M.nx)
            
            # duS += (_duS[1:] + _duS[:-1])/2
            _aduS = np.zeros(M.nx)
            _aduS[1:] += aduS/2
            _aduS[:-1] += aduS/2
            
        #     _duS = M.g/(w**2-fS**2)*(\
        #   dhbcx[j,0,0,i] * (w*kx*np.cos(phase_v)+fS*ky*np.sin(phase_v))\
        # +dkxy_v * hbcx[j,0,0,i]*(w*kx*np.sin(phase_v)-fS*ky*np.cos(phase_v))\
        # +dkx * hbcx[j,0,0,i]*w*np.cos(phase_v)\
        # +dky * hbcx[j,0,0,i]*fS*np.sin(phase_v) +\
        #   dhbcx[j,0,1,i] * (w*kx*np.sin(phase_v)-fS*ky*np.cos(phase_v))\
        # -dkxy_v * hbcx[j,0,1,i]*(w*kx*np.cos(phase_v)+fS*ky*np.sin(phase_v))\
        # +dkx * hbcx[j,0,1,i]*w*np.sin(phase_v)\
        # -dky * hbcx[j,0,1,i]*fS*np.cos(phase_v)) 
            adhbcx1d_incr[j,0,0,i] += M.g/(w**2-fS**2)*\
                (w*kx*np.cos(phase_v)+fS*ky*np.sin(phase_v)) * _aduS
            adkxy_v += M.g/(w**2-fS**2)*\
                hbcx[j,0,0,i]*(w*kx*np.sin(phase_v)-fS*ky*np.cos(phase_v)) * _aduS
            adkx += M.g/(w**2-fS**2)*\
                hbcx[j,0,0,i]*w*np.cos(phase_v) * _aduS
            adky += M.g/(w**2-fS**2)*\
                hbcx[j,0,0,i]*fS*np.sin(phase_v) * _aduS
            
            adhbcx1d_incr[j,0,1,i] += M.g/(w**2-fS**2)*\
                (w*kx*np.sin(phase_v)-fS*ky*np.cos(phase_v)) * _aduS
            adkxy_v += -M.g/(w**2-fS**2)*\
                hbcx[j,0,1,i]*(w*kx*np.cos(phase_v)+fS*ky*np.sin(phase_v)) * _aduS
            adkx += M.g/(w**2-fS**2)*\
                hbcx[j,0,1,i]*w*np.sin(phase_v) * _aduS
            adky += -M.g/(w**2-fS**2)*\
                hbcx[j,0,1,i]*fS*np.cos(phase_v) * _aduS
        
        #     dvS += M.g/(w**2-fS**2)*(\
        #   dhbcx[j,0,0,i] * (w*ky*np.cos(phase_v)-fS*kx*np.sin(phase_v))\
        # +dkxy_v * hbcx[j,0,0,i]*(w*ky*np.sin(phase_v)+fS*kx*np.cos(phase_v))\
        # +dky * hbcx[j,0,0,i]*w*np.cos(phase_v)\
        # -dkx * hbcx[j,0,0,i]*fS*np.sin(phase_v) +\
        #   dhbcx[j,0,1,i] * (w*ky*np.sin(phase_v)+fS*kx*np.cos(phase_v))\
        # -dkxy_v * hbcx[j,0,1,i]*(w*ky*np.cos(phase_v)-fS*kx*np.sin(phase_v))\
        # +dky * hbcx[j,0,1,i]*w*np.sin(phase_v)\
        # +dkx * hbcx[j,0,1,i]*fS*np.cos(phase_v)
        # )  
            adhbcx1d_incr[j,0,0,i] += M.g/(w**2-fS**2)*\
                (w*ky*np.cos(phase_v)-fS*kx*np.sin(phase_v)) * advS
            adkxy_v += M.g/(w**2-fS**2)*\
                hbcx[j,0,0,i]*(w*ky*np.sin(phase_v)+fS*kx*np.cos(phase_v)) * advS
            adky += M.g/(w**2-fS**2)*\
                hbcx[j,0,0,i]*w*np.cos(phase_v) * advS
            adkx += -M.g/(w**2-fS**2)*\
                hbcx[j,0,0,i]*fS*np.sin(phase_v) * advS
            adhbcx1d_incr[j,0,1,i] += M.g/(w**2-fS**2)*\
                (w*ky*np.sin(phase_v)+fS*kx*np.cos(phase_v)) * advS
            adkxy_v += -M.g/(w**2-fS**2)*\
                hbcx[j,0,1,i]*(w*ky*np.cos(phase_v)-fS*kx*np.sin(phase_v)) * advS
            adky += M.g/(w**2-fS**2)*\
                hbcx[j,0,1,i]*w*np.sin(phase_v) * advS
            adkx += M.g/(w**2-fS**2)*\
                hbcx[j,0,1,i]*fS*np.cos(phase_v) * advS
                
            # dhS += dhbcx[j,0,0,i] * np.cos(phase_h) +\
            #     dkxy_h * hbcx[j,0,0,i]*np.sin(phase_h) +\
            #           dhbcx[j,0,1,i] * np.sin(phase_h) -\
            #     dkxy_h * hbcx[j,0,1,i]*np.cos(phase_h)
            adhbcx1d_incr[j,0,0,i] += np.cos(phase_h) * adhS
            adkxy_h += hbcx[j,0,0,i]*np.sin(phase_h) * adhS
            adhbcx1d_incr[j,0,1,i] += np.sin(phase_h) * adhS
            adkxy_h += -hbcx[j,0,1,i]*np.cos(phase_h) * adhS
            
            # dkxy_v = -dHeS/HeS * kxy_v/2
            adHeS += -adkxy_v/HeS * kxy_v/2
            
            # dkxy_h = -dHeS/HeS * kxy_h/2
            adHeS += -adkxy_h/HeS * kxy_h/2
            
            # dky = -dHeS/HeS * ky/2
            adHeS += -adky/HeS * ky/2
            
            # dkx = -dHeS/HeS * kx/2
            adHeS += -adkx/HeS * kx/2
            
            ###############################################################
            # North
            ###############################################################
            fN = (M.f[-1,:]+M.f[-2,:])/2
            HeN = (He[-1,:]+He[-2,:])/2
            k = np.sqrt((w**2-fN**2)/(M.g*HeN))
            kx = np.sin(theta) * k
            ky = -np.cos(theta) * k
            kxy_h = kx*M.X[-1,:] + ky*M.Y[-1,:]
            kxy_v = kx*M.Xv[-1,:] + ky*M.Yv[-1,:]
            phase_h =  w*t0 - kxy_h
            phase_v =  w*t0 - kxy_v
            
            
            adkx = np.zeros(M.nx)
            adky = np.zeros(M.nx)
            adkxy_v = np.zeros(M.nx)
            adkxy_h = np.zeros(M.nx)
            
            # duN += (_duN[1:] + _duN[:-1])/2
            _aduN = np.zeros(M.nx)
            _aduN[1:] += aduN/2
            _aduN[:-1] += aduN/2
            
        #     _duN = M.g/(w**2-fN**2)*(\
        #   dhbcx[j,1,0,i] * (w*kx*np.cos(phase_v)+fN*ky*np.sin(phase_v))\
        # +dkxy_v * hbcx[j,1,0,i]*(w*kx*np.sin(phase_v)-fN*ky*np.cos(phase_v))\
        # +dkx * hbcx[j,1,0,i]*w*np.cos(phase_v)\
        # +dky * hbcx[j,1,0,i]*fN*np.sin(phase_v) +\
        #   dhbcx[j,1,1,i] * (w*kx*np.sin(phase_v)-fN*ky*np.cos(phase_v))\
        # -dkxy_v * hbcx[j,1,1,i]*(w*kx*np.cos(phase_v)+fN*ky*np.sin(phase_v))\
        # +dkx * hbcx[j,1,1,i]*w*np.sin(phase_v)\
        # -dky * hbcx[j,1,1,i]*fN*np.cos(phase_v)
        # ) 
            adhbcx1d_incr[j,1,0,i] += M.g/(w**2-fN**2)*\
                (w*kx*np.cos(phase_v)+fN*ky*np.sin(phase_v)) * _aduN
            adkxy_v += M.g/(w**2-fN**2)*\
                hbcx[j,1,0,i]*(w*kx*np.sin(phase_v)-fN*ky*np.cos(phase_v)) * _aduN
            adkx += M.g/(w**2-fN**2)*\
                hbcx[j,1,0,i]*w*np.cos(phase_v) * _aduN
            adky += M.g/(w**2-fN**2)*\
                hbcx[j,1,0,i]*fN*np.sin(phase_v) * _aduN
            
            adhbcx1d_incr[j,1,1,i] += M.g/(w**2-fN**2)*\
                (w*kx*np.sin(phase_v)-fN*ky*np.cos(phase_v)) * _aduN
            adkxy_v += -M.g/(w**2-fN**2)*\
                hbcx[j,1,1,i]*(w*kx*np.cos(phase_v)+fN*ky*np.sin(phase_v)) * _aduN
            adkx += M.g/(w**2-fN**2)*\
                hbcx[j,1,1,i]*w*np.sin(phase_v) * _aduN
            adky += -M.g/(w**2-fN**2)*\
                hbcx[j,1,1,i]*fN*np.cos(phase_v) * _aduN
        
        #     dvN += M.g/(w**2-fN**2)*(\
        #   dhbcx[j,1,0,i] * (w*ky*np.cos(phase_v)-fN*kx*np.sin(phase_v))\
        # +dkxy_v * hbcx[j,1,0,i]*(w*ky*np.sin(phase_v)+fN*kx*np.cos(phase_v))\
        # +dky * hbcx[j,1,0,i]*w*np.cos(phase_v)\
        # -dkx * hbcx[j,1,0,i]*fN*np.sin(phase_v) +\
        #   dhbcx[j,1,1,i] * (w*ky*np.sin(phase_v)+fN*kx*np.cos(phase_v))\
        # -dkxy_v * hbcx[j,1,1,i]*(w*ky*np.cos(phase_v)-fN*kx*np.sin(phase_v))\
        # +dky * hbcx[j,1,1,i]*w*np.sin(phase_v)\
        # +dkx * hbcx[j,1,1,i]*fN*np.cos(phase_v)
        # ) 
            adhbcx1d_incr[j,1,0,i] += M.g/(w**2-fN**2)*\
                (w*ky*np.cos(phase_v)-fN*kx*np.sin(phase_v)) * advN
            adkxy_v += M.g/(w**2-fN**2)*\
                hbcx[j,1,0,i]*(w*ky*np.sin(phase_v)+fN*kx*np.cos(phase_v)) * advN
            adky += M.g/(w**2-fN**2)*\
                hbcx[j,1,0,i]*w*np.cos(phase_v) * advN
            adkx += -M.g/(w**2-fN**2)*\
                hbcx[j,1,0,i]*fN*np.sin(phase_v) * advN
            adhbcx1d_incr[j,1,1,i] += M.g/(w**2-fN**2)*\
                (w*ky*np.sin(phase_v)+fN*kx*np.cos(phase_v)) * advN
            adkxy_v += -M.g/(w**2-fN**2)*\
                hbcx[j,1,1,i]*(w*ky*np.cos(phase_v)-fN*kx*np.sin(phase_v)) * advN
            adky += M.g/(w**2-fN**2)*\
                hbcx[j,1,1,i]*w*np.sin(phase_v) * advN
            adkx += M.g/(w**2-fN**2)*\
                hbcx[j,1,1,i]*fN*np.cos(phase_v) * advN
                
            # dhN += dhbcx[j,1,0,i] * np.cos(phase_h) +\
                # dkxy_h * hbcx[j,1,0,i]*np.sin(phase_h) +\
                #       dhbcx[j,1,1,i] * np.sin(phase_h) -\
                # dkxy_h * hbcx[j,1,1,i]*np.cos(phase_h)
            adhbcx1d_incr[j,1,0,i] += np.cos(phase_h) * adhN
            adkxy_h += hbcx[j,1,0,i]*np.sin(phase_h) * adhN
            adhbcx1d_incr[j,1,1,i] += np.sin(phase_h) * adhN
            adkxy_h += -hbcx[j,1,1,i]*np.cos(phase_h) * adhN
            
            # dkxy_v = -dHeN/HeN * kxy_v/2
            adHeN += -adkxy_v/HeN * kxy_v/2
            
            # dkxy_h = -dHeN/HeN * kxy_h/2
            adHeN += -adkxy_h/HeN * kxy_h/2
            
            # dky = -dHeN/HeN * ky/2
            adHeN += -adky/HeN * ky/2
            
            # dkx = -dHeN/HeN * kx/2
            adHeN += -adkx/HeN * kx/2
            
            ###############################################################
            # West
            ###############################################################
            fW = (M.f[:,0]+M.f[:,1])/2
            HeW = (He[:,0]+He[:,1])/2
            k = np.sqrt((w**2-fW**2)/(M.g*HeW))
            kx = np.cos(theta) * k
            ky = np.sin(theta) * k
            kxy_h = kx*M.X[:,0] + ky*M.Y[:,0]
            kxy_u = kx*M.Xu[:,0] + ky*M.Yu[:,0]
            phase_h =  w*t0 - kxy_h
            phase_u =  w*t0 - kxy_u
            
            adkx = np.zeros(M.ny)
            adky = np.zeros(M.ny)
            adkxy_u = np.zeros(M.ny)
            adkxy_h = np.zeros(M.ny)
            
            # dvW += (_dvW[1:] + _dvW[:-1])/2
            _advW = np.zeros(M.ny)
            _advW[1:] += advW/2
            _advW[:-1] += advW/2
            
        #     _dvW = M.g/(w**2-fW**2)*(\
        #   dhbcy[j,0,0,i] * (w*ky*np.cos(phase_u)-fW*kx*np.sin(phase_u))\
        # +dkxy_u * hbcy[j,0,0,i]*(w*ky*np.sin(phase_u)+fW*kx*np.cos(phase_u))\
        # +dky * hbcy[j,0,0,i]*w*np.cos(phase_u)\
        # -dkx * hbcy[j,0,0,i]*fW*np.sin(phase_u) +\
        #   dhbcy[j,0,1,i] * (w*ky*np.sin(phase_u)+fW*kx*np.cos(phase_u))\
        # -dkxy_u * hbcy[j,0,1,i]*(w*ky*np.cos(phase_u)-fW*kx*np.sin(phase_u))\
        # +dky * hbcy[j,0,1,i]*w*np.sin(phase_u)\
        # +dkx * hbcy[j,0,1,i]*fW*np.cos(phase_u)
        # )  
            adhbcy1d_incr[j,0,0,i] += M.g/(w**2-fW**2)*\
                (w*ky*np.cos(phase_u)-fW*kx*np.sin(phase_u)) * _advW
            adkxy_u += M.g/(w**2-fW**2)*\
                hbcy[j,0,0,i]*(w*ky*np.sin(phase_u)+fW*kx*np.cos(phase_u)) * _advW
            adky += M.g/(w**2-fW**2)*\
                hbcy[j,0,0,i]*w*np.cos(phase_u) * _advW
            adkx += -M.g/(w**2-fW**2)*\
                hbcy[j,0,0,i]*fW*np.sin(phase_u) * _advW
            adhbcy1d_incr[j,0,1,i] += M.g/(w**2-fW**2)*\
                (w*ky*np.sin(phase_u)+fW*kx*np.cos(phase_u)) * _advW
            adkxy_u += -M.g/(w**2-fW**2)*\
                hbcy[j,0,1,i]*(w*ky*np.cos(phase_u)-fW*kx*np.sin(phase_u)) * _advW
            adky += M.g/(w**2-fW**2)*\
                hbcy[j,0,1,i]*w*np.sin(phase_u) * _advW
            adkx += M.g/(w**2-fW**2)*\
                hbcy[j,0,1,i]*fW*np.cos(phase_u) * _advW
            
        
        #     duW += M.g/(w**2-fW**2)*(\
        #   dhbcy[j,0,0,i] * (w*kx*np.cos(phase_u)+fW*ky*np.sin(phase_u))\
        # +dkxy_u * hbcy[j,0,0,i]*(w*kx*np.sin(phase_u)-fW*ky*np.cos(phase_u))\
        # +dkx * hbcy[j,0,0,i]*w*np.cos(phase_u)\
        # +dky * hbcy[j,0,0,i]*fW*np.sin(phase_u) +\
        #   dhbcy[j,0,1,i] * (w*kx*np.sin(phase_u)-fW*ky*np.cos(phase_u))\
        # -dkxy_u * hbcy[j,0,1,i]*(w*kx*np.cos(phase_u)+fW*ky*np.sin(phase_u))\
        # +dkx * hbcy[j,0,1,i]*w*np.sin(phase_u)\
        # -dky * hbcy[j,0,1,i]*fW*np.cos(phase_u)
        # ) 
            adhbcy1d_incr[j,0,0,i] += M.g/(w**2-fW**2)*\
                (w*kx*np.cos(phase_u)+fW*ky*np.sin(phase_u)) * aduW
            adkxy_u += M.g/(w**2-fW**2)*\
                hbcy[j,0,0,i]*(w*kx*np.sin(phase_u)-fW*ky*np.cos(phase_u)) * aduW
            adkx += M.g/(w**2-fW**2)*\
                hbcy[j,0,0,i]*w*np.cos(phase_u) * aduW
            adky += M.g/(w**2-fW**2)*\
                hbcy[j,0,0,i]*fW*np.sin(phase_u) * aduW
            adhbcy1d_incr[j,0,1,i] += M.g/(w**2-fW**2)*\
                (w*kx*np.sin(phase_u)-fW*ky*np.cos(phase_u)) * aduW
            adkxy_u += -M.g/(w**2-fW**2)*\
                hbcy[j,0,1,i]*(w*kx*np.cos(phase_u)+fW*ky*np.sin(phase_u)) * aduW
            adkx += M.g/(w**2-fW**2)*\
                hbcy[j,0,1,i]*w*np.sin(phase_u) * aduW
            adky += -M.g/(w**2-fW**2)*\
                hbcy[j,0,1,i]*fW*np.cos(phase_u) * aduW
                
            # dhW += dhbcy[j,0,0,i] * np.cos(phase_h) +\
            #         dkxy_h * hbcy[j,0,0,i]*np.sin(phase_h) +\
            #           dhbcy[j,0,1,i] * np.sin(phase_h) -\
            #         dkxy_h * hbcy[j,0,1,i]*np.cos(phase_h)
            adhbcy1d_incr[j,0,0,i] += np.cos(phase_h) * adhW
            adkxy_h += hbcy[j,0,0,i]*np.sin(phase_h) * adhW
            adhbcy1d_incr[j,0,1,i] += np.sin(phase_h) * adhW
            adkxy_h += -hbcy[j,0,1,i]*np.cos(phase_h) * adhW
            
            # dkxy_u = -dHeW/HeW * kxy_u/2
            adHeW += -adkxy_u/HeW * kxy_u/2
            
            # dkxy_h = -dHeW/HeW * kxy_h/2
            adHeW += -adkxy_h/HeW * kxy_h/2
            
            # dky = -dHeW/HeW * ky/2
            adHeW += -adky/HeW * ky/2
            
            # dkx = -dHeW/HeW * kx/2
            adHeW += -adkx/HeW * kx/2
            
            ###############################################################
            # East
            ###############################################################
            fE = (M.f[:,-1]+M.f[:,-2])/2
            HeE = (He[:,-1]+He[:,-2])/2
            k = np.sqrt((w**2-fE**2)/(M.g*HeE))
            kx = -np.cos(theta) * k
            ky = np.sin(theta) * k
            kxy_h = kx*M.X[:,-1] + ky*M.Y[:,-1]
            kxy_u = kx*M.Xu[:,-1] + ky*M.Yu[:,-1]
            phase_h =  w*t0 - kxy_h
            phase_u =  w*t0 - kxy_u
            
            adkx = np.zeros(M.ny)
            adky = np.zeros(M.ny)
            adkxy_u = np.zeros(M.ny)
            adkxy_h = np.zeros(M.ny)
            
            # dvE += (_dvE[1:] + _dvE[:-1])/2
            _advE = np.zeros(M.ny)
            _advE[1:] += advE/2
            _advE[:-1] += advE/2
            
        #     _dvE = M.g/(w**2-fE**2)*(\
        #   dhbcy[j,1,0,i] * (w*ky*np.cos(phase_u)-fE*kx*np.sin(phase_u))\
        # +dkxy_u * hbcy[j,1,0,i]*(w*ky*np.sin(phase_u)+fE*kx*np.cos(phase_u))\
        # +dky * hbcy[j,1,0,i]*w*np.cos(phase_u)\
        # -dkx * hbcy[j,1,0,i]*fE*np.sin(phase_u) +\
        #   dhbcy[j,1,1,i] * (w*ky*np.sin(phase_u)+fE*kx*np.cos(phase_u))\
        # -dkxy_u * hbcy[j,1,1,i]*(w*ky*np.cos(phase_u)-fE*kx*np.sin(phase_u))\
        # +dky * hbcy[j,1,1,i]*w*np.sin(phase_u)\
        # +dkx * hbcy[j,1,1,i]*fE*np.cos(phase_u)
        # )  
            adhbcy1d_incr[j,1,0,i] += M.g/(w**2-fE**2)*\
                (w*ky*np.cos(phase_u)-fE*kx*np.sin(phase_u)) * _advE
            adkxy_u += M.g/(w**2-fE**2)*\
                hbcy[j,1,0,i]*(w*ky*np.sin(phase_u)+fE*kx*np.cos(phase_u)) * _advE
            adky += M.g/(w**2-fE**2)*\
                hbcy[j,1,0,i]*w*np.cos(phase_u) * _advE
            adkx += -M.g/(w**2-fE**2)*\
                hbcy[j,1,0,i]*fE*np.sin(phase_u) * _advE
            adhbcy1d_incr[j,1,1,i] += M.g/(w**2-fE**2)*\
                (w*ky*np.sin(phase_u)+fE*kx*np.cos(phase_u)) * _advE
            adkxy_u += -M.g/(w**2-fE**2)*\
                hbcy[j,1,1,i]*(w*ky*np.cos(phase_u)-fE*kx*np.sin(phase_u)) * _advE
            adky += M.g/(w**2-fE**2)*\
                hbcy[j,1,1,i]*w*np.sin(phase_u) * _advE
            adkx += M.g/(w**2-fE**2)*\
                hbcy[j,1,1,i]*fE*np.cos(phase_u) * _advE
            
        
        #     duE += M.g/(w**2-fE**2)*(\
        #   dhbcy[j,1,0,i] * (w*kx*np.cos(phase_u)+fE*ky*np.sin(phase_u))\
        # +dkxy_u * hbcy[j,1,0,i]*(w*kx*np.sin(phase_u)-fE*ky*np.cos(phase_u))\
        # +dkx * hbcy[j,1,0,i]*w*np.cos(phase_u)\
        # +dky * hbcy[j,1,0,i]*fE*np.sin(phase_u) +\
        #   dhbcy[j,1,1,i] * (w*kx*np.sin(phase_u)-fE*ky*np.cos(phase_u))\
        # -dkxy_u * hbcy[j,1,1,i]*(w*kx*np.cos(phase_u)+fE*ky*np.sin(phase_u))\
        # +dkx * hbcy[j,1,1,i]*w*np.sin(phase_u)\
        # -dky * hbcy[j,1,1,i]*fE*np.cos(phase_u)
        # ) 
            adhbcy1d_incr[j,1,0,i] += M.g/(w**2-fE**2)*\
                (w*kx*np.cos(phase_u)+fE*ky*np.sin(phase_u)) * aduE
            adkxy_u += M.g/(w**2-fE**2)*\
                hbcy[j,1,0,i]*(w*kx*np.sin(phase_u)-fE*ky*np.cos(phase_u)) * aduE
            adkx += M.g/(w**2-fE**2)*\
                hbcy[j,1,0,i]*w*np.cos(phase_u) * aduE
            adky += M.g/(w**2-fE**2)*\
                hbcy[j,1,0,i]*fE*np.sin(phase_u) * aduE
            adhbcy1d_incr[j,1,1,i] += M.g/(w**2-fE**2)*\
                (w*kx*np.sin(phase_u)-fE*ky*np.cos(phase_u)) * aduE
            adkxy_u += -M.g/(w**2-fE**2)*\
                hbcy[j,1,1,i]*(w*kx*np.cos(phase_u)+fE*ky*np.sin(phase_u)) * aduE
            adkx += M.g/(w**2-fE**2)*\
                hbcy[j,1,1,i]*w*np.sin(phase_u) * aduE
            adky += -M.g/(w**2-fE**2)*\
                hbcy[j,1,1,i]*fE*np.cos(phase_u) * aduE
                
            # dhE += dhbcy[j,1,0,i] * np.cos(phase_h) +\
            #         dkxy_h * hbcy[j,1,0,i]*np.sin(phase_h) +\
            #           dhbcy[j,1,1,i] * np.sin(phase_h) -\
            #         dkxy_h * hbcy[j,1,1,i]*np.cos(phase_h)
            adhbcy1d_incr[j,1,0,i] += np.cos(phase_h) * adhE
            adkxy_h += hbcy[j,1,0,i]*np.sin(phase_h) * adhE
            adhbcy1d_incr[j,1,1,i] += np.sin(phase_h) * adhE
            adkxy_h += -hbcy[j,1,1,i]*np.cos(phase_h) * adhE
            
            # dkxy_u = -dHeE/HeE * kxy_u/2
            adHeE += -adkxy_u/HeE * kxy_u/2
            
            # dkxy_h = -dHeE/HeE * kxy_h/2
            adHeE += -adkxy_h/HeE * kxy_h/2
            
            # dky = -dHeE/HeE * ky/2
            adHeE += -adky/HeE * ky/2
            
            # dkx = -dHeE/HeE * kx/2
            adHeE += -adkx/HeE * kx/2
            
    
    adHe2d_incr[0,:] += adHeS/2
    adHe2d_incr[1,:] += adHeS/2
    adHe2d_incr[-1,:] += adHeN/2
    adHe2d_incr[-2,:] += adHeN/2
    adHe2d_incr[:,0] += adHeW/2
    adHe2d_incr[:,1] += adHeW/2
    adHe2d_incr[:,-1] += adHeE/2
    adHe2d_incr[:,-2] += adHeE/2
    
    
    
    return adHe2d_incr,adhbcx1d_incr,adhbcy1d_incr

    
    
def obcs_adj(M,t,
             adu0,adv0,adh0,
             u0,v0,h0,He,hbcx,hbcy):
    
    if M.bc=='1d':
        t += M.dt
        
    # Init 
    adu_incr = np.zeros(M.Xu.shape)
    adv_incr = np.zeros(M.Xv.shape)
    adh_incr = np.zeros(M.X.shape)
    adHe2d_incr = np.zeros(M.X.shape)
    adhbcx1d_incr = np.zeros([M.omegas.size,2,2,M.bc_theta.size,M.nx])
    adhbcy1d_incr = np.zeros([M.omegas.size,2,2,M.bc_theta.size,M.ny])
    
    aduS,advS,adhS,aduN,advN,adhN,aduW,advW,adhW,aduE,advE,adhE = \
        update_borders_adj(M,adu0,adv0,adh0)
    
    #######################################################################
    # South
    #######################################################################
    # 0. Init boundary variables
    fS = (M.f[0,:]+M.f[1,:])/2
    HeS = (He[0,:]+He[1,:])/2
    cS = np.sqrt(M.g*HeS)
    if M.bc=='1d':
        cS *= M.dt/(M.Y[1,:]-M.Y[0,:])
        
    # 1. w1   
    w1_ext = np.zeros(M.nx)
    for j,w in enumerate(M.omegas):
        k = np.sqrt((w**2-fS**2)/(M.g*HeS))
        for i,theta in enumerate(M.bc_theta):
            kx = np.sin(theta) * k
            ky = np.cos(theta) * k
            kxy = kx*M.Xv[0,:] + ky*M.Yv[0,:]
            h_ext = hbcx[j,0,0,i]* np.cos(w*t-kxy)  +\
                    hbcx[j,0,1,i]* np.sin(w*t-kxy) 
            v_ext = M.g/(w**2-fS**2)*(\
                hbcx[j,0,0,i]* (w*ky*np.cos(w*t-kxy) \
                            - fS*kx*np.sin(w*t-kxy)
                                ) +\
                hbcx[j,0,1,i]* (w*ky*np.sin(w*t-kxy) \
                            + fS*kx*np.cos(w*t-kxy)
                                )
                    )
            w1_ext += v_ext + np.sqrt(M.g/HeS) * h_ext      
    if M.bc=='1d':
        w1S = +w1_ext
    elif M.bc=='2d':
        # dw1dy0
        w10  = v0[0,:] + np.sqrt(M.g/HeS)* (h0[0,:]+h0[1,:])/2
        w10_ = (v0[0,:]+v0[1,:])/2 + np.sqrt(M.g/HeS)* h0[1,:]
        _w10 = +w1_ext
        dw1dy0 = (w10_ - _w10)/M.dy
        # dudx0
        dudx0 = np.zeros(M.nx)
        dudx0[1:-1] = ((u0[0,1:] + u0[1,1:] - u0[0,:-1] - u0[1,:-1])/2)/M.dx
        dudx0[0] = dudx0[1]
        dudx0[-1] = dudx0[-2]
        # w1S
        w1S = w10 - M.dt*cS* (dw1dy0 + dudx0)
        
    # 3. w3
    if M.bc=='1d':
        _vS = (1-3/2*cS)* v0[0,:] + cS/2* (4*v0[1,:] - v0[2,:])
        _hS = (1/2+cS)* h0[1,:] + (1/2-cS)* h0[0,:]
        w3S = _vS - np.sqrt(M.g/HeS) * _hS
    elif M.bc=='2d':
        w30   = v0[0,:] - np.sqrt(M.g/HeS)* (h0[0,:]+h0[1,:])/2
        w30_  = (v0[0,:]+v0[1,:])/2  - np.sqrt(M.g/HeS)* h0[1,:]
        w30__ = v0[1,:] - np.sqrt(M.g/HeS)* (h0[1,:]+h0[2,:])/2
        dw3dy0 =  -(3*w30 - 4*w30_ + w30__)/(M.dy/2)
        w3S = w30 + M.dt*cS* (dw3dy0 + dudx0) 
    
    # Adj 4. Values on BC
    #duS = dw2S
    adw2S = +aduS
    aduS = 0
    #dvS = (dw1S + dw3S)/2 
    adw3S = advS/2
    adw1S = advS/2
    advS = 0
    # dhS = 1/2 * (np.sqrt(HeS/M.g) * (dw1S - dw3S) +\
    #             1/(2*np.sqrt(HeS*M.g))*(w1S-w3S) * dHeS)
    adw1S += 1/2 * np.sqrt(HeS/M.g) * adhS
    adw3S += -1/2 * np.sqrt(HeS/M.g) * adhS   
    adHeS = 1/2/(2*np.sqrt(HeS*M.g))*(w1S-w3S) * adhS          
    
    # Adj 3. w3
    if M.bc=='1d':
        #dw3S = _dvS - np.sqrt(M.g/HeS) * _dhS +\
        #    1/2*np.sqrt(M.g/HeS**3)*_hS * dHeS 
        _advS = +adw3S
        _adhS = -np.sqrt(M.g/HeS) * adw3S
        adHeS += 1/2*np.sqrt(M.g/HeS**3)*_hS * adw3S
        #_dhS = h0[1,:] * dcS - h0[0,:] * dcS + (1/2+cS)* dh0[1,:] +\
        #    (1/2-cS)* dh0[0,:]
        adcS = (h0[1,:]-h0[0,:]) * _adhS 
        adh_incr[1,:] += (1/2+cS) * _adhS
        adh_incr[0,:] += (1/2-cS) * _adhS
        #_dvS = -3/2*v0[0,:]* dcS + (4*v0[1,:] - v0[2,:])/2 * dcS + \
        #    (1-3/2*cS)* dv0[0,:] + cS/2* (4*dv0[1,:] - dv0[2,:])
        adcS += (2*v0[1,:]-v0[2,:]/2-3/2*v0[0,:]) * _advS
        adv_incr[0,:] += (1-3/2*cS) * _advS
        adv_incr[1,:] += 2*cS * _advS
        adv_incr[2,:] += -cS/2 * _advS
    elif M.bc=='2d':
        #dw3S = dw30 + M.dt*(dcS* (dw3dy0 + dudx0) + cS* (ddw3dy0 + ddudx0))
        adw30 = +adw3S
        adcS = M.dt*(dw3dy0 + dudx0) * adw3S
        addw3dy0 = M.dt*cS * adw3S
        addudx0 = M.dt*cS * adw3S
        #ddw3dy0 =  -(3*dw30 - 4*dw30_ + dw30__)/(M.dy/2)
        adw30  += -3*addw3dy0/(M.dy/2)
        adw30_  =  4*addw3dy0/(M.dy/2)
        adw30__ =   -addw3dy0/(M.dy/2)
        #dw30__ = dv0[1,:] + (1/2)*np.sqrt(M.g/HeS**3)*(h0[1,:]+h0[2,:])/2 * dHeS -\
        #    np.sqrt(M.g/HeS)* (dh0[1,:]+dh0[2,:])/2
        adv_incr[1,:] += adw30__
        adHeS += (1/2)*np.sqrt(M.g/HeS**3)*(h0[1,:]+h0[2,:])/2 * adw30__
        adh_incr[1,:] += -np.sqrt(M.g/HeS)/2 * adw30__
        adh_incr[2,:] += -np.sqrt(M.g/HeS)/2 * adw30__
        # dw30_  = (dv0[0,:]+dv0[1,:])/2 + (1/2)*np.sqrt(M.g/HeS**3)* h0[1,:] * dHeS -\
        #     np.sqrt(M.g/HeS)* dh0[1,:]
        adv_incr[0,:] += adw30_/2
        adv_incr[1,:] += adw30_/2
        adHeS += (1/2)*np.sqrt(M.g/HeS**3)* h0[1,:] * adw30_
        adh_incr[1,:] += -np.sqrt(M.g/HeS)* adw30_
        # dw30 = dv0[0,:] + (1/2)*np.sqrt(M.g/HeS**3)*(h0[0,:]+h0[1,:])/2 * dHeS -\
        #     np.sqrt(M.g/HeS)* (dh0[0,:]+dh0[1,:])/2
        adv_incr[0,:] += adw30
        adHeS += (1/2)*np.sqrt(M.g/HeS**3)*(h0[0,:]+h0[1,:])/2 * adw30
        adh_incr[1,:] += -np.sqrt(M.g/HeS)* adw30/2
        adh_incr[0,:] += -np.sqrt(M.g/HeS)* adw30/2
        
    # Adj 2. w2
    if M.bc=='1d':
        #dw2S = dw20
        adw20 = +adw2S
    elif M.bc=='2d':
        #dw2S = dw20 - M.dt*M.g* ddhdx0 
        adw20 = +adw2S
        addhdx0 = -M.dt*M.g* adw2S
        #ddhdx0 = ((dh0[0,1:]+dh0[1,1:]-dh0[0,:-1]-dh0[1,:-1])/2)/M.dx
        adh_incr[0,1:] += addhdx0/(2*M.dx)
        adh_incr[1,1:] += addhdx0/(2*M.dx)
        adh_incr[0,:-1] += -addhdx0/(2*M.dx)
        adh_incr[1,:-1] += -addhdx0/(2*M.dx)
    #dw20 = (du0[0,:] + du0[1,:])/2
    adu_incr[0,:] += adw20/2
    adu_incr[1,:] += adw20/2
    
    # Adj 1. w1
    if M.bc=='1d':
        #dw1S = dw1_ext
        adw1_ext = +adw1S
    elif M.bc=='2d':
        #dw1S = dw10 - M.dt*(dcS* (dw1dy0 + dudx0) + cS* (ddw1dy0 + ddudx0))
        adw10 = +adw1S
        adcS += -M.dt*(dw1dy0 + dudx0) * adw1S
        addw1dy0 = -M.dt*cS * adw1S
        addudx0 += -M.dt*cS * adw1S
        #ddudx0[-1] = ddudx0[-2]
        addudx0[-2] += addudx0[-1]
        #ddudx0[0] = ddudx0[1]
        addudx0[1] += addudx0[0]
        #ddudx0[1:-1] = ((du0[0,1:] + du0[1,1:] - du0[0,:-1] - du0[1,:-1])/2)/M.dx
        adu_incr[0,1:] += addudx0[1:-1]/(2*M.dx)
        adu_incr[1,1:] += addudx0[1:-1]/(2*M.dx)
        adu_incr[0,:-1] += -addudx0[1:-1]/(2*M.dx)
        adu_incr[1,:-1] += -addudx0[1:-1]/(2*M.dx)
        #ddw1dy0 = (dw10_ - _dw10)/M.dy
        adw10_ = addw1dy0/M.dy
        _adw10 = -addw1dy0/M.dy
        #_dw10 = dw1_ext
        adw1_ext = +_adw10
        # dw10_ = (dv0[0,:]+dv0[1,:])/2 - (1/2)*np.sqrt(M.g/HeS**3)*h0[1,:] * dHeS +\
        #     np.sqrt(M.g/HeS)* dh0[1,:]
        adv_incr[0,:] += adw10_/2
        adv_incr[1,:] += adw10_/2
        adHeS += -(1/2)*np.sqrt(M.g/HeS**3)*h0[1,:] * adw10_
        adh_incr[1,:] += np.sqrt(M.g/HeS)* adw10_
        # dw10 = dv0[0,:] - (1/2)*np.sqrt(M.g/HeS**3)*(h0[0,:]+h0[1,:])/2 * dHeS +\
        #     np.sqrt(M.g/HeS)* (dh0[0,:]+dh0[1,:])/2
        adv_incr[0,:] += adw10
        adHeS += -(1/2)*np.sqrt(M.g/HeS**3)*(h0[0,:]+h0[1,:])/2 * adw10
        adh_incr[0,:] += np.sqrt(M.g/HeS)/2* adw10
        adh_incr[1,:] += np.sqrt(M.g/HeS)/2* adw10
        
    for j,w in enumerate(M.omegas):
        k = np.sqrt((w**2-fS**2)/(M.g*HeS))
        #adh_ext = np.zeros(M.nx)
        #adv_ext = np.zeros(M.nx)
        for i,theta in enumerate(M.bc_theta):
            kx = np.sin(theta) * k
            ky = np.cos(theta) * k
            kxy = kx*M.Xv[0,:] + ky*M.Yv[0,:]
            h_ext = hbcx[j,0,0,i]* np.cos(w*t-kxy)  +\
                    hbcx[j,0,1,i]* np.sin(w*t-kxy) 
            #dw1_ext += dv_ext + np.sqrt(M.g/HeS) * dh_ext - \
            #     1/2*np.sqrt(M.g/HeS**3)*h_ext * dHeS 
            adh_ext = np.sqrt(M.g/HeS) * adw1_ext
            adv_ext = +adw1_ext
            adHeS += -1/2*np.sqrt(M.g/HeS**3)*h_ext * adw1_ext
            #dv_ext = M.g/(w**2-fS**2)*(\
        #   dhbcx[j,0,0,i] * (w*ky*np.cos(w*t-kxy)-fS*kx*np.sin(w*t-kxy))\
        # +dkxy * hbcx[j,0,0,i]*(w*ky*np.sin(w*t-kxy)+fS*kx*np.cos(w*t-kxy))\
        # +dky * hbcx[j,0,0,i]*w*np.cos(w*t-kxy)\
        # -dkx * hbcx[j,0,0,i]*fS*np.sin(w*t-kxy) +\
        #   dhbcx[j,0,1,i] * (w*ky*np.sin(w*t-kxy)+fS*kx*np.cos(w*t-kxy))\
        # -dkxy * hbcx[j,0,1,i]*(w*ky*np.cos(w*t-kxy)-fS*kx*np.sin(w*t-kxy))\
        # +dky * hbcx[j,0,1,i]*w*np.sin(w*t-kxy)\
        # +dkx * hbcx[j,0,1,i]*fS*np.cos(w*t-kxy))
            adhbcx1d_incr[j,0,0,i] += M.g/(w**2-fS**2)*\
                (w*ky*np.cos(w*t-kxy)-fS*kx*np.sin(w*t-kxy)) * adv_ext
            adkxy = M.g/(w**2-fS**2)*\
                hbcx[j,0,0,i]*(w*ky*np.sin(w*t-kxy)+fS*kx*np.cos(w*t-kxy)) * adv_ext
            adky = M.g/(w**2-fS**2)*\
                hbcx[j,0,0,i]*w*np.cos(w*t-kxy) * adv_ext
            adkx = -M.g/(w**2-fS**2)*\
                hbcx[j,0,0,i]*fS*np.sin(w*t-kxy) * adv_ext
            adhbcx1d_incr[j,0,1,i] += M.g/(w**2-fS**2)*\
                (w*ky*np.sin(w*t-kxy)+fS*kx*np.cos(w*t-kxy)) * adv_ext
            adkxy += -M.g/(w**2-fS**2)*\
                hbcx[j,0,1,i]*(w*ky*np.cos(w*t-kxy)-fS*kx*np.sin(w*t-kxy)) * adv_ext
            adky += M.g/(w**2-fS**2)*\
                hbcx[j,0,1,i]*w*np.sin(w*t-kxy) * adv_ext
            adkx += M.g/(w**2-fS**2)*\
                hbcx[j,0,1,i]*fS*np.cos(w*t-kxy) * adv_ext
            # dh_ext = dhbcx[j,0,0,i] * np.cos(w*t-kxy) +\
            #     dkxy * hbcx[j,0,0,i]*np.sin(w*t-kxy) +\
            #          dhbcx[j,0,1,i] * np.sin(w*t-kxy) -\
            #     dkxy * hbcx[j,0,1,i]*np.cos(w*t-kxy)
            adhbcx1d_incr[j,0,0,i] += np.cos(w*t-kxy) * adh_ext
            adkxy += hbcx[j,0,0,i]*np.sin(w*t-kxy) * adh_ext
            adhbcx1d_incr[j,0,1,i] += np.sin(w*t-kxy) * adh_ext
            adkxy += -hbcx[j,0,1,i]*np.cos(w*t-kxy) * adh_ext
            #dkxy = -dHeS/HeS * kxy/2
            adHeS += -adkxy/HeS * kxy/2
            #dky = -dHeS/HeS * ky/2
            adHeS += -adky/HeS * ky/2
            #dkx = -dHeS/HeS * kx/2
            adHeS += -adkx/HeS * kx/2
            

        
    # Adj 0. Init boundary variables
    adHeS += cS/HeS/2 * adcS
    adHe2d_incr[0,:] += adHeS/2
    adHe2d_incr[1,:] += adHeS/2
    
    #######################################################################
    # North
    #######################################################################
    fN = (M.f[-1,:]+M.f[-2,:])/2
    HeN = (He[-1,:]+He[-2,:])/2
    cN = np.sqrt(M.g*HeN)
    if M.bc=='1d':
        cN *= M.dt/(M.Y[-1,:]-M.Y[-2,:])
    # 1. w1   
    w1_ext = np.zeros(M.nx)
    for j,w in enumerate(M.omegas):
        k = np.sqrt((w**2-fN**2)/(M.g*HeN))
        for i,theta in enumerate(M.bc_theta):
            kx = np.sin(theta) * k
            ky = -np.cos(theta) * k
            kxy = kx*M.Xv[-1,:] + ky*M.Yv[-1,:]
            h_ext = hbcx[j,1,0,i]* np.cos(w*t-kxy)+\
                    hbcx[j,1,1,i]* np.sin(w*t-kxy) 
            v_ext = M.g/(w**2-fN**2)*(\
                hbcx[j,1,0,i]* (w*ky*np.cos(w*t-kxy) \
                            - fN*kx*np.sin(w*t-kxy)
                                ) +\
                hbcx[j,1,1,i]* (w*ky*np.sin(w*t-kxy) \
                            + fN*kx*np.cos(w*t-kxy)
                                )
                    )
            w1_ext += v_ext - np.sqrt(M.g/HeN) * h_ext      
    if M.bc=='1d':
        w1N = w1_ext
    elif M.bc=='2d':
        w10  = v0[-1,:] - np.sqrt(M.g/HeN)* (h0[-1,:]+h0[-2,:])/2
        w10_ = (v0[-1,:]+v0[-2,:])/2 - np.sqrt(M.g/HeN)* h0[-2,:]
        _w10 = w1_ext
        dw1dy0 = (_w10 - w10_)/M.dy
        dudx0 = np.zeros(M.nx)
        dudx0[1:-1] = ((u0[-1,1:] + u0[-2,1:] - u0[-1,:-1] - u0[-2,:-1])/2)/M.dx
        dudx0[0] = dudx0[1]
        dudx0[-1] = dudx0[-2]
        w1N = w10 + M.dt*cN* (dw1dy0 + dudx0) 
    # 3. w3
    if M.bc=='1d':   
        _vN = (1-3/2*cN)* v0[-1,:] + cN/2* (4*v0[-2,:] - v0[-3,:])
        _hN = (1/2+cN)* h0[-2,:] + (1/2-cN)* h0[-1,:]
        w3N = _vN + np.sqrt(M.g/HeN) * _hN
    elif M.bc=='2d':
        w30   = v0[-1,:] + np.sqrt(M.g/HeN)* (h0[-1,:]+h0[-2,:])/2
        w30_  = (v0[-1,:]+v0[-2,:])/2 + np.sqrt(M.g/HeN)* h0[-2,:]
        w30__ = v0[-2,:] + np.sqrt(M.g/HeN)* (h0[-2,:]+h0[-3,:])/2
        dw3dy0 =  (3*w30 - 4*w30_ + w30__)/(M.dy/2)
        w3N = w30 - M.dt*cN* (dw3dy0 + dudx0) 
    
    # Adj 4. Values on BC
    #duN = dw2N
    adw2N = +aduN
    aduN = 0
    #dvN = (dw1N + dw3N)/2 
    adw3N = advN/2
    adw1N = advN/2
    advN = 0
    # dhN = 1/2 * (np.sqrt(HeN/M.g) * (dw3N - dw1N) +\
    #             1/(2*np.sqrt(HeN*M.g))*(w3N-w1N) * dHeN)
    adw1N += -1/2 * np.sqrt(HeN/M.g) * adhN
    adw3N += 1/2 * np.sqrt(HeN/M.g) * adhN   
    adHeN = 1/2/(2*np.sqrt(HeN*M.g))*(w3N-w1N) * adhN   
       
    # Adj 3. w3
    if M.bc=='1d':
        #dw3N = _dvN + np.sqrt(M.g/HeN) * _dhN -\
        #    1/2*np.sqrt(M.g/HeN**3)*_hN * dHeN 
        _advN = +adw3N
        _adhN = np.sqrt(M.g/HeN) * adw3N
        adHeN += -1/2*np.sqrt(M.g/HeN**3)*_hN * adw3N
        #_dhN = h0[-2,:] * dcN - h0[-1,:] * dcN + (1/2+cN)* dh0[-2,:] +\
        #    (1/2-cN)* dh0[-1,:]
        adcN = (h0[-2,:]-h0[-1,:]) * _adhN 
        adh_incr[-2,:] += (1/2+cN) * _adhN
        adh_incr[-1,:] += (1/2-cN) * _adhN
        #_dvN = -3/2*v0[-1,:]* dcN + (4*v0[-2,:] - v0[-3,:])/2 * dcN + \
        #    (1-3/2*cN)* dv0[-1,:] + cN/2* (4*dv0[-2,:] - dv0[-3,:])
        adcN += (2*v0[-2,:]-v0[-3,:]/2-3/2*v0[-1,:]) * _advN
        adv_incr[-1,:] += (1-3/2*cN) * _advN
        adv_incr[-2,:] += 2*cN * _advN
        adv_incr[-3,:] += -cN/2 * _advN
    elif M.bc=='2d':
        #dw3N = dw30 - M.dt*(dcN* (dw3dy0 + dudx0) + cN* (ddw3dy0 + ddudx0))
        adw30 = +adw3N
        adcN = -M.dt*(dw3dy0 + dudx0) * adw3N
        addw3dy0 = -M.dt*cN * adw3N
        addudx0 = -M.dt*cN * adw3N
        #ddw3dy0 =  (3*dw30 - 4*dw30_ + dw30__)/(M.dy/2)
        adw30  += 3*addw3dy0/(M.dy/2)
        adw30_  = -4*addw3dy0/(M.dy/2)
        adw30__ =    addw3dy0/(M.dy/2)
        # dw30__ = dv0[-2,:] - (1/2)*np.sqrt(M.g/HeN**3)*(h0[-2,:]+h0[-3,:])/2 * dHeN +\
        #     np.sqrt(M.g/HeN)* (dh0[-2,:]+dh0[-3,:])/2
        adv_incr[-2,:] += adw30__
        adHeN += -(1/2)*np.sqrt(M.g/HeN**3)*(h0[-2,:]+h0[-3,:])/2 * adw30__
        adh_incr[-2,:] += np.sqrt(M.g/HeN)/2 * adw30__
        adh_incr[-3,:] += np.sqrt(M.g/HeN)/2 * adw30__
        # dw30_  = (dv0[-1,:]+dv0[-2,:])/2 - (1/2)*np.sqrt(M.g/HeN**3)* h0[-2,:] * dHeN +\
        #     np.sqrt(M.g/HeN)* dh0[-2,:]
        adv_incr[-1,:] += adw30_/2
        adv_incr[-2,:] += adw30_/2
        adHeN += -(1/2)*np.sqrt(M.g/HeN**3)* h0[-2,:] * adw30_
        adh_incr[-2,:] += np.sqrt(M.g/HeN)* adw30_
        # dw30 = dv0[-1,:] - (1/2)*np.sqrt(M.g/HeN**3)*(h0[-1,:]+h0[-2,:])/2 * dHeN +\
        #     np.sqrt(M.g/HeN)* (dh0[-1,:]+dh0[-2,:])/2
        adv_incr[-1,:] += adw30
        adHeN += -(1/2)*np.sqrt(M.g/HeN**3)*(h0[-1,:]+h0[-2,:])/2 * adw30
        adh_incr[-2,:] += np.sqrt(M.g/HeN)* adw30/2
        adh_incr[-1,:] += np.sqrt(M.g/HeN)* adw30/2
    
    # Adj 2. w2
    if M.bc=='1d':
        #dw2N = dw20
        adw20 = +adw2N
    elif M.bc=='2d':
        #dw2N = dw20 - M.dt*M.g*ddhdx0 
        adw20 = +adw2N
        addhdx0 = -M.dt*M.g* adw2N
        #ddhdx0 = ((dh0[-1,1:]+dh0[-2,1:]-dh0[-1,:-1]-dh0[-2,:-1])/2)/M.dx
        adh_incr[-1,1:] += addhdx0/(2*M.dx)
        adh_incr[-2,1:] += addhdx0/(2*M.dx)
        adh_incr[-1,:-1] += -addhdx0/(2*M.dx)
        adh_incr[-2,:-1] += -addhdx0/(2*M.dx)
    #dw20 = (du0[-1,:] + du0[-2,:])/2
    adu_incr[-1,:] += adw20/2
    adu_incr[-2,:] += adw20/2
    
    # Adj 1. w1
    if M.bc=='1d':
        #dw1N = dw1_ext
        adw1_ext = +adw1N
    elif M.bc=='2d':
        #dw1N = dw10 + M.dt*(dcN* (dw1dy0 + dudx0) + cN* (ddw1dy0 + ddudx0))
        adw10 = +adw1N
        adcN += M.dt*(dw1dy0 + dudx0) * adw1N
        addw1dy0 = M.dt*cN * adw1N
        addudx0 += M.dt*cN * adw1N
        #ddudx0[-1] = ddudx0[-2]
        addudx0[-2] += addudx0[-1]
        #ddudx0[0] = ddudx0[1]
        addudx0[1] += addudx0[0]
        #ddudx0[1:-1] = ((du0[-1,1:] + du0[-2,1:] - du0[-1,:-1] - du0[-2,:-1])/2)/M.dx
        adu_incr[-1,1:] += addudx0[1:-1]/(2*M.dx)
        adu_incr[-2,1:] += addudx0[1:-1]/(2*M.dx)
        adu_incr[-1,:-1] += -addudx0[1:-1]/(2*M.dx)
        adu_incr[-2,:-1] += -addudx0[1:-1]/(2*M.dx)
        #ddw1dy0 = (_dw10 - dw10_)/M.dy
        _adw10 = addw1dy0/M.dy
        adw10_ = -addw1dy0/M.dy
        #_dw10 = dw1_ext
        adw1_ext = +_adw10
        # dw10_ = (dv0[-1,:]+dv0[-2,:])/2 + (1/2)*np.sqrt(M.g/HeN**3)*h0[-2,:] * dHeN -\
        #     np.sqrt(M.g/HeN)* dh0[-2,:]
        adv_incr[-1,:] += adw10_/2
        adv_incr[-2,:] += adw10_/2
        adHeN += (1/2)*np.sqrt(M.g/HeN**3)*h0[-2,:] * adw10_
        adh_incr[-2,:] += -np.sqrt(M.g/HeN)* adw10_
        # dw10 = dv0[-1,:] + (1/2)*np.sqrt(M.g/HeN**3)*(h0[-1,:]+h0[-2,:])/2 * dHeN -\
        #     np.sqrt(M.g/HeN)* (dh0[-1,:]+dh0[-2,:])/2
        adv_incr[-1,:] += adw10
        adHeN += (1/2)*np.sqrt(M.g/HeN**3)*(h0[-1,:]+h0[-2,:])/2 * adw10
        adh_incr[-1,:] += -np.sqrt(M.g/HeN)/2* adw10
        adh_incr[-2,:] += -np.sqrt(M.g/HeN)/2* adw10
    for j,w in enumerate(M.omegas):
        k = np.sqrt((w**2-fN**2)/(M.g*HeN))
        #adh_ext = np.zeros(M.nx)
        #adv_ext = np.zeros(M.nx)
        for i,theta in enumerate(M.bc_theta):
            kx = np.sin(theta) * k
            ky = -np.cos(theta) * k
            kxy = kx*M.Xv[-1,:] + ky*M.Yv[-1,:]
            h_ext = hbcx[j,1,0,i]* np.cos(w*t-kxy)+\
                    hbcx[j,1,1,i]* np.sin(w*t-kxy) 
            #dw1_ext += dv_ext - np.sqrt(M.g/HeN) * dh_ext + \
            #     1/2*np.sqrt(M.g/HeN**3)*h_ext * dHeN 
            adh_ext = -np.sqrt(M.g/HeN) * adw1_ext
            adv_ext = +adw1_ext
            adHeN += 1/2*np.sqrt(M.g/HeN**3)*h_ext * adw1_ext
            # dv_ext = M.g/(w**2-fN**2)*(\
        #   dhbcx[j,1,0,i] * (w*ky*np.cos(w*t-kxy)-fN*kx*np.sin(w*t-kxy))\
        # +dkxy * hbcx[j,1,0,i]*(w*ky*np.sin(w*t-kxy)+fN*kx*np.cos(w*t-kxy))\
        # +dky * hbcx[j,1,0,i]*w*np.cos(w*t-kxy)\
        # -dkx * hbcx[j,1,0,i]*fN*np.sin(w*t-kxy) +\
        #   dhbcx[j,1,1,i] * (w*ky*np.sin(w*t-kxy)+fN*kx*np.cos(w*t-kxy))\
        # -dkxy * hbcx[j,1,1,i]*(w*ky*np.cos(w*t-kxy)-fN*kx*np.sin(w*t-kxy))\
        # +dky * hbcx[j,1,1,i]*w*np.sin(w*t-kxy)\
        # +dkx * hbcx[j,1,1,i]*fN*np.cos(w*t-kxy))
            adhbcx1d_incr[j,1,0,i] += M.g/(w**2-fN**2)*\
                (w*ky*np.cos(w*t-kxy)-fN*kx*np.sin(w*t-kxy)) * adv_ext
            adkxy = M.g/(w**2-fN**2)*\
                hbcx[j,1,0,i]*(w*ky*np.sin(w*t-kxy)+fN*kx*np.cos(w*t-kxy)) * adv_ext
            adky = M.g/(w**2-fN**2)*\
                hbcx[j,1,0,i]*w*np.cos(w*t-kxy) * adv_ext
            adkx = -M.g/(w**2-fN**2)*\
                hbcx[j,1,0,i]*fN*np.sin(w*t-kxy) * adv_ext
            adhbcx1d_incr[j,1,1,i] += M.g/(w**2-fN**2)*\
                (w*ky*np.sin(w*t-kxy)+fN*kx*np.cos(w*t-kxy)) * adv_ext
            adkxy += -M.g/(w**2-fN**2)*\
                hbcx[j,1,1,i]*(w*ky*np.cos(w*t-kxy)-fN*kx*np.sin(w*t-kxy)) * adv_ext
            adky += M.g/(w**2-fN**2)*\
                hbcx[j,1,1,i]*w*np.sin(w*t-kxy) * adv_ext
            adkx += M.g/(w**2-fN**2)*\
                hbcx[j,1,1,i]*fN*np.cos(w*t-kxy) * adv_ext
            # dh_ext = dhbcx[j,1,0,i] * np.cos(w*t-kxy) +\
            #         dkxy * hbcx[j,1,0,i]*np.sin(w*t-kxy) +\
            #          dhbcx[j,1,1,i] * np.sin(w*t-kxy) -\
            #         dkxy * hbcx[j,1,1,i]*np.cos(w*t-kxy)
            adhbcx1d_incr[j,1,0,i] += np.cos(w*t-kxy) * adh_ext
            adkxy += hbcx[j,1,0,i]*np.sin(w*t-kxy) * adh_ext
            adhbcx1d_incr[j,1,1,i] += np.sin(w*t-kxy) * adh_ext
            adkxy += -hbcx[j,1,1,i]*np.cos(w*t-kxy) * adh_ext
            #dkxy = -dHeN/HeN * kxy/2
            adHeN += -adkxy/HeN * kxy/2
            #dky = -dHeN/HeN * ky/2
            adHeN += -adky/HeN * ky/2
            #dkx = -dHeN/HeN * kx/2
            adHeN += -adkx/HeN * kx/2
            
    # Adj 0. Init boundary variables
    adHeN += cN/HeN/2 * adcN
    adHe2d_incr[-1,:] += adHeN/2
    adHe2d_incr[-2,:] += adHeN/2
    
    #######################################################################
    # West
    #######################################################################
    # 0. Init boundary variables
    fW = (M.f[:,0]+M.f[:,1])/2
    HeW = (He[:,0]+He[:,1])/2
    cW = np.sqrt(M.g*HeW)
    if M.bc=='1d':
        cW *= M.dt/(M.X[:,1]-M.X[:,0])
    # 1. w1   
    w1_ext = np.zeros(M.ny)
    for j,w in enumerate(M.omegas):
        k = np.sqrt((w**2-fW**2)/(M.g*HeW))
        for i,theta in enumerate(M.bc_theta):
            kx = np.cos(theta) * k
            ky = np.sin(theta) * k
            kxy = kx*M.Xu[:,0] + ky*M.Yu[:,0]
            h_ext = hbcy[j,0,0,i]*np.cos(w*t-kxy) +\
                    hbcy[j,0,1,i]*np.sin(w*t-kxy)
            u_ext = M.g/(w**2-fW**2)*(\
                hbcy[j,0,0,i]*(w*kx*np.cos(w*t-kxy) \
                          + fW*ky*np.sin(w*t-kxy)
                              ) +\
                hbcy[j,0,1,i]*(w*kx*np.sin(w*t-kxy) \
                          - fW*ky*np.cos(w*t-kxy)
                              )
                    )
            w1_ext += u_ext + np.sqrt(M.g/HeW) * h_ext     
    if M.bc=='1d':   
        w1W = w1_ext
    elif M.bc=='2d':
        w10  = u0[:,0] + np.sqrt(M.g/HeW)* (h0[:,0]+h0[:,1])/2
        w10_ = (u0[:,0]+u0[:,1])/2 + np.sqrt(M.g/HeW)* h0[:,1]
        _w10 = w1_ext
        dw1dx0 = (w10_ - _w10)/M.dx
        dvdy0 = np.zeros(M.ny)
        dvdy0[1:-1] = ((v0[1:,0] + v0[1:,1] - v0[:-1,0] - v0[:-1,1])/2)/M.dy
        dvdy0[0] = dvdy0[1]
        dvdy0[-1] = dvdy0[-2]
        w1W = w10 - M.dt*cW* (dw1dx0 + dvdy0) 
    # 3. w3
    if M.bc=='1d':   
        _uW = (1-3/2*cW)* u0[:,0] + cW/2* (4*u0[:,1]-u0[:,2]) 
        _hW = (1/2+cW)*h0[:,1] + (1/2-cW)*h0[:,0]
        w3W = _uW - np.sqrt(M.g/HeW)* _hW
    elif M.bc=='2d':
        w30   = u0[:,0] - np.sqrt(M.g/HeW)* (h0[:,0]+h0[:,1])/2
        w30_  = (u0[:,0]+u0[:,1])/2 - np.sqrt(M.g/HeW)* h0[:,1]
        w30__ = u0[:,1] - np.sqrt(M.g/HeW)* (h0[:,1]+h0[:,2])/2
        dw3dx0 = -(3*w30 - 4*w30_ + w30__)/(M.dx/2)
        w3W = w30 + M.dt*cW* (dw3dx0 + dvdy0)
    
    # Adj 4. Values on BC
    #dvW = dw2W
    adw2W = +advW
    advW = 0
    #duW = (dw1W + dw3W)/2 
    adw3W = aduW/2
    adw1W = aduW/2
    aduW = 0
    # dhW = 1/2 * (np.sqrt(HeW/M.g) * (dw1W - dw3W) +\
    #             1/(2*np.sqrt(HeW*M.g))*(w1W-w3W) * dHeW)
    adw1W += 1/2 * np.sqrt(HeW/M.g) * adhW
    adw3W += -1/2 * np.sqrt(HeW/M.g) * adhW   
    adHeW = 1/2/(2*np.sqrt(HeW*M.g))*(w1W-w3W) * adhW 
         
    # Adj 3. w3
    if M.bc=='1d':   
        #dw3W = _dvW - np.sqrt(M.g/HeW) * _dhW +\
        #    1/2*np.sqrt(M.g/HeW**3)*_hW * dHeW 
        _aduW = +adw3W
        _adhW = -np.sqrt(M.g/HeW) * adw3W
        adHeW += 1/2*np.sqrt(M.g/HeW**3)*_hW * adw3W
        #_dhW = h0[:,1] * dcW - h0[:,0] * dcW + (1/2+cW)* dh0[:,1] +\
        #    (1/2-cW)* dh0[:,0]
        adcW = (h0[:,1]-h0[:,0]) * _adhW 
        adh_incr[:,1] += (1/2+cW) * _adhW
        adh_incr[:,0] += (1/2-cW) * _adhW
        #_dvW = -3/2*u0[:,0]* dcW + (4*u0[:,1] - u0[:,2])/2 * dcW + \
        #    (1-3/2*cW)* du0[:,0] + cW/2* (4*du0[:,1] - du0[:,2])
        adcW += (2*u0[:,1]-u0[:,2]/2-3/2*u0[:,0]) * _aduW
        adu_incr[:,0] += (1-3/2*cW) * _aduW
        adu_incr[:,1] += 2*cW * _aduW
        adu_incr[:,2] += -cW/2 * _aduW
    elif M.bc=='2d':
        #dw3W = dw30 + M.dt*(dcW* (dw3dx0 + dvdy0) + cW* (ddw3dx0 + ddvdy0))
        adw30 = +adw3W
        adcW = M.dt*(dw3dx0 + dvdy0) * adw3W
        addw3dx0 = M.dt*cW * adw3W
        addvdy0 = M.dt*cW * adw3W
        #ddw3dx0 = -(3*dw30 - 4*dw30_ + dw30__)/(M.dx/2)
        adw30  += -3*addw3dx0/(M.dx/2)
        adw30_  =  4*addw3dx0/(M.dx/2)
        adw30__ =   -addw3dx0/(M.dx/2)
        # dw30__= du0[:,1] + (1/2)*np.sqrt(M.g/HeW**3)*(h0[:,1]+h0[:,2])/2 * dHeW -\
        #     np.sqrt(M.g/HeW)* (dh0[:,1]+dh0[:,2])/2
        adu_incr[:,1] += adw30__
        adHeW += (1/2)*np.sqrt(M.g/HeW**3)*(h0[:,1]+h0[:,2])/2 * adw30__
        adh_incr[:,1] += -np.sqrt(M.g/HeW)/2 * adw30__
        adh_incr[:,2] += -np.sqrt(M.g/HeW)/2 * adw30__
        # dw30_ = (du0[:,0]+du0[:,1])/2 + (1/2)*np.sqrt(M.g/HeW**3)*h0[:,1] * dHeW -\
        #    np.sqrt(M.g/HeW)* dh0[:,1]
        adu_incr[:,0] += adw30_/2
        adu_incr[:,1] += adw30_/2
        adHeW += (1/2)*np.sqrt(M.g/HeW**3)* h0[:,1] * adw30_
        adh_incr[:,1] += -np.sqrt(M.g/HeW)* adw30_
        # dw30  = du0[:,0] + (1/2)*np.sqrt(M.g/HeW**3)*(h0[:,0]+h0[:,1])/2 * dHeW -\
        #     np.sqrt(M.g/HeW)* (dh0[:,0]+dh0[:,1])/2
        adu_incr[:,0] += adw30
        adHeW += (1/2)*np.sqrt(M.g/HeW**3)*(h0[:,0]+h0[:,1])/2 * adw30
        adh_incr[:,1] += -np.sqrt(M.g/HeW)* adw30/2
        adh_incr[:,0] += -np.sqrt(M.g/HeW)* adw30/2
        
    # Adj 2. w2
    if M.bc=='1d':
        # dw2W = dw20
        adw20 = +adw2W
    elif M.bc=='2d':
        # dw2W = dw20 - M.dt*M.g * ddhdy0  
        adw20 = +adw2W
        addhdy0 = -M.dt*M.g* adw2W
        # ddhdy0 = ((dh0[1:,0]+dh0[1:,1]-dh0[:-1,0]-dh0[:-1,1])/2)/M.dy
        adh_incr[1:,0] += addhdy0/(2*M.dy)
        adh_incr[1:,1] += addhdy0/(2*M.dy)
        adh_incr[:-1,0] += -addhdy0/(2*M.dy)
        adh_incr[:-1,1] += -addhdy0/(2*M.dy)
    # dw20 = (dv0[:,0] + dv0[:,1])/2
    adv_incr[:,0] += adw20/2
    adv_incr[:,1] += adw20/2
    
    # Adj 1. w1
    if M.bc=='1d':   
        adw1_ext = +adw1W
    elif M.bc=='2d':
        # dw1W = dw10 - M.dt*(dcW* (dw1dx0 + dvdy0) + cW* (ddw1dx0 + ddvdy0))
        adw10 = +adw1W
        adcW += -M.dt*(dw1dx0 + dvdy0) * adw1W
        addw1dx0 = -M.dt*cW * adw1W
        addvdy0 += -M.dt*cW * adw1W
        #ddvdy0[-1] = ddvdy0[-2]
        addvdy0[-2] += addvdy0[-1]
        #ddvdy0[0] = ddvdy0[1]
        addvdy0[1] += addvdy0[0]
        #ddvdy0[1:-1] = ((dv0[1:,0] + dv0[1:,1] - dv0[:-1,0] - dv0[:-1,1])/2)/M.dy
        adv_incr[1:,0] += addvdy0[1:-1]/(2*M.dy)
        adv_incr[1:,1] += addvdy0[1:-1]/(2*M.dy)
        adv_incr[:-1,0] += -addvdy0[1:-1]/(2*M.dy)
        adv_incr[:-1,1] += -addvdy0[1:-1]/(2*M.dy)
        #ddw1dx0 = (dw10_ - _dw10)/M.dx
        adw10_ = addw1dx0/M.dx
        _adw10 = -addw1dx0/M.dx
        #_dw10 = dw1_ext
        adw1_ext = +_adw10
        # dw10_ = (du0[:,0]+du0[:,1])/2 - (1/2)*np.sqrt(M.g/HeW**3)*h0[:,1] * dHeW +\
        #     np.sqrt(M.g/HeW)* dh0[:,1]
        adu_incr[:,0] += adw10_/2
        adu_incr[:,1] += adw10_/2
        adHeW += -(1/2)*np.sqrt(M.g/HeW**3)*h0[:,1] * adw10_
        adh_incr[:,1] += np.sqrt(M.g/HeW)* adw10_
        # dw10  = du0[:,0] - (1/2)*np.sqrt(M.g/HeW**3)*(h0[:,0]+h0[:,1])/2 * dHeW +\
        #     np.sqrt(M.g/HeW)* (dh0[:,0]+dh0[:,1])/2 
        adu_incr[:,0] += adw10
        adHeW += -(1/2)*np.sqrt(M.g/HeW**3)*(h0[:,0]+h0[:,1])/2 * adw10
        adh_incr[:,0] += np.sqrt(M.g/HeW)/2* adw10
        adh_incr[:,1] += np.sqrt(M.g/HeW)/2* adw10
    for j,w in enumerate(M.omegas):
        k = np.sqrt((w**2-fW**2)/(M.g*HeW))
        #adh_ext = np.zeros(M.nx)
        #adv_ext = np.zeros(M.nx)
        for i,theta in enumerate(M.bc_theta):
            kx = np.cos(theta) * k
            ky = np.sin(theta) * k
            kxy = kx*M.Xu[:,0] + ky*M.Yu[:,0]
            h_ext = hbcy[j,0,0,i]*np.cos(w*t-kxy) +\
                    hbcy[j,0,1,i]*np.sin(w*t-kxy)
            #dw1_ext += du_ext + np.sqrt(M.g/HeW) * dh_ext - \
            #     1/2*np.sqrt(M.g/HeW**3)*h_ext * dHeW 
            adh_ext = np.sqrt(M.g/HeW) * adw1_ext
            adu_ext = +adw1_ext
            adHeW += -1/2*np.sqrt(M.g/HeW**3)*h_ext * adw1_ext
        #     du_ext = M.g/(w**2-fW**2)*(\
        #   dhbcy[j,0,0,i] * (w*kx*np.cos(w*t-kxy)+fW*ky*np.sin(w*t-kxy))\
        # +dkxy * hbcy[j,0,0,i]*(w*kx*np.sin(w*t-kxy)-fW*ky*np.cos(w*t-kxy))\
        # +dkx * hbcy[j,0,0,i]*w*np.cos(w*t-kxy)\
        # +dky * hbcy[j,0,0,i]*fW*np.sin(w*t-kxy) +\
        #   dhbcy[j,0,1,i] * (w*kx*np.sin(w*t-kxy)-fW*ky*np.cos(w*t-kxy))\
        # -dkxy * hbcy[j,0,1,i]*(w*kx*np.cos(w*t-kxy)+fW*ky*np.sin(w*t-kxy))\
        # +dkx * hbcy[j,0,1,i]*w*np.sin(w*t-kxy)\
        # -dky * hbcy[j,0,1,i]*fW*np.cos(w*t-kxy)) 
            adhbcy1d_incr[j,0,0,i] += M.g/(w**2-fW**2)*\
                (w*kx*np.cos(w*t-kxy)+fW*ky*np.sin(w*t-kxy)) * adu_ext
            adkxy = M.g/(w**2-fW**2)*\
                hbcy[j,0,0,i]*(w*kx*np.sin(w*t-kxy)-fW*ky*np.cos(w*t-kxy)) * adu_ext
            adkx = M.g/(w**2-fW**2)*\
                hbcy[j,0,0,i]*w*np.cos(w*t-kxy) * adu_ext
            adky = M.g/(w**2-fW**2)*\
                hbcy[j,0,0,i]*fW*np.sin(w*t-kxy) * adu_ext
            adhbcy1d_incr[j,0,1,i] += M.g/(w**2-fW**2)*\
                (w*kx*np.sin(w*t-kxy)-fW*ky*np.cos(w*t-kxy)) * adu_ext
            adkxy += -M.g/(w**2-fW**2)*\
                hbcy[j,0,1,i]*(w*kx*np.cos(w*t-kxy)+fW*ky*np.sin(w*t-kxy)) * adu_ext
            adkx += M.g/(w**2-fW**2)*\
                hbcy[j,0,1,i]*w*np.sin(w*t-kxy) * adu_ext
            adky += -M.g/(w**2-fW**2)*\
                hbcy[j,0,1,i]*fW*np.cos(w*t-kxy) * adu_ext
            # dh_ext = dhbcy[j,0,0,i] * np.cos(w*t-kxy) +\
            #         dkxy * hbcy[j,0,0,i]*np.sin(w*t-kxy) +\
            #          dhbcy[j,0,1,i] * np.sin(w*t-kxy) -\
            #         dkxy * hbcy[j,0,1,i]*np.cos(w*t-kxy)
            adhbcy1d_incr[j,0,0,i] += np.cos(w*t-kxy) * adh_ext
            adkxy += hbcy[j,0,0,i]*np.sin(w*t-kxy) * adh_ext
            adhbcy1d_incr[j,0,1,i] += np.sin(w*t-kxy) * adh_ext
            adkxy += -hbcy[j,0,1,i]*np.cos(w*t-kxy) * adh_ext
            #dkxy = -dHeW/HeW * kxy/2
            adHeW += -adkxy/HeW * kxy/2
            #dky = -dHeW/HeW * ky/2
            adHeW += -adky/HeW * ky/2
            #dkx = -dHeW/HeW * kx/2
            adHeW += -adkx/HeW * kx/2
            
    # Adj 0. Init boundary variables
    adHeW += cW/HeW/2 * adcW
    adHe2d_incr[:,0] += adHeW/2
    adHe2d_incr[:,1] += adHeW/2
    
    #######################################################################
    # East
    #######################################################################
    # 0. Init boundary variables
    fE = (M.f[:,-1]+M.f[:,-2])/2
    HeE = (He[:,-1]+He[:,-2])/2
    cE = np.sqrt(M.g*HeE)
    if M.bc=='1d':
        cE *= M.dt/(M.X[:,-1]-M.X[:,-2])
    # 1. w1   
    w1_ext = np.zeros(M.ny)
    for j,w in enumerate(M.omegas):
        k = np.sqrt((w**2-fE**2)/(M.g*HeE))
        for i,theta in enumerate(M.bc_theta):
            kx = -np.cos(theta) * k
            ky = np.sin(theta) * k
            kxy = kx*M.Xu[:,-1] + ky*M.Yu[:,-1]
            h_ext = hbcy[j,1,0,i]*np.cos(w*t-kxy) +\
                    hbcy[j,1,1,i]*np.sin(w*t-kxy)
            u_ext = M.g/(w**2-fE**2)*(\
                hbcy[j,1,0,i]* (w*kx*np.cos(w*t-kxy) \
                            + fE*ky*np.sin(w*t-kxy)
                                ) +\
                hbcy[j,1,1,i]*(w*kx*np.sin(w*t-kxy) \
                          - fE*ky*np.cos(w*t-kxy)
                              )
                    )
            w1_ext += u_ext - np.sqrt(M.g/HeE) * h_ext      
    if M.bc=='1d':   
        w1E = w1_ext
    elif M.bc=='2d':
        w10  = u0[:,-1] - np.sqrt(M.g/HeE)* (h0[:,-1]+h0[:,-2])/2
        w10_ = (u0[:,-1]+u0[:,-2])/2 - np.sqrt(M.g/HeE)* h0[:,-2]
        _w10 = w1_ext
        dw1dx0 = (_w10 - w10_)/M.dx
        dvdy0 = np.zeros(M.ny)
        dvdy0[1:-1] = ((v0[1:,-1] + v0[1:,-2] - v0[:-1,-1] - v0[:-1,-2])/2)/M.dy
        dvdy0[0] = dvdy0[1]
        dvdy0[-1] = dvdy0[-2]
        w1E = w10 + M.dt*cE* (dw1dx0 + dvdy0) 
    # 3. w3
    if M.bc=='1d':   
        _uE = (1-3/2*cE)* u0[:,-1] + cE/2* (4*u0[:,-2]-u0[:,-3])
        _hE = ((1/2+cE)*h0[:,-2] + (1/2-cE)*h0[:,-1])
        w3E = _uE + np.sqrt(M.g/HeE)* _hE 
    elif M.bc=='2d':
        w30   = u0[:,-1] + np.sqrt(M.g/HeE)* (h0[:,-1]+h0[:,-2])/2
        w30_  = (u0[:,-1]+u0[:,-2])/2 + np.sqrt(M.g/HeE)* h0[:,-2]
        w30__ = u0[:,-2] + np.sqrt(M.g/HeE)* (h0[:,-2]+h0[:,-3])/2
        dw3dx0 =  (3*w30 - 4*w30_ + w30__)/(M.dx/2)
        w3E = w30 - M.dt*cE* (dw3dx0 + dvdy0) 
    
    # Adj 4. Values on BC
    #dvE = dw2E
    adw2E = +advE
    advE = 0
    #duE = (dw1E + dw3E)/2 
    adw3E = aduE/2
    adw1E = aduE/2
    aduE = 0
    # dhE = 1/2 * (np.sqrt(HeE/M.g) * (dw3E - dw1E) +\
    #             1/(2*np.sqrt(HeE*M.g))*(w3E-w1E) * dHeE)
    adw1E += -1/2 * np.sqrt(HeE/M.g) * adhE
    adw3E += 1/2 * np.sqrt(HeE/M.g) * adhE   
    adHeE = 1/2/(2*np.sqrt(HeE*M.g))*(w3E-w1E) * adhE   
       
    # Adj 3. w3
    if M.bc=='1d': 
        #dw3E = _dvE + np.sqrt(M.g/HeE) * _dhE -\
        #    1/2*np.sqrt(M.g/HeE**3)*_hE * dHeE 
        _aduE = +adw3E
        _adhE = np.sqrt(M.g/HeE) * adw3E
        adHeE += -1/2*np.sqrt(M.g/HeE**3)*_hE * adw3E
        #_dhE = h0[:,-2] * dcE - h0[:,-1] * dcE + (1/2+cE)* dh0[:,-2] +\
        #    (1/2-cE)* dh0[:,-1]
        adcE = (h0[:,-2]-h0[:,-1]) * _adhE 
        adh_incr[:,-2] += (1/2+cE) * _adhE
        adh_incr[:,-1] += (1/2-cE) * _adhE
        #_dvE = -3/2*u0[:,-1]* dcE + (4*u0[:,-2] - u0[:,-3])/2 * dcE + \
        #    (1-3/2*cE)* du0[:,-1] + cE/2* (4*du0[:,-2] - du0[:,-3])
        adcE += (2*u0[:,-2]-u0[:,-3]/2-3/2*u0[:,-1]) * _aduE
        adu_incr[:,-1] += (1-3/2*cE) * _aduE
        adu_incr[:,-2] += 2*cE * _aduE
        adu_incr[:,-3] += -cE/2 * _aduE
    elif M.bc=='2d':
        #dw3E = dw30 - M.dt*(dcE* (dw3dx0 + dvdy0) + cE* (ddw3dx0 + ddvdy0))
        adw30 = +adw3E
        adcE = -M.dt*(dw3dx0 + dvdy0) * adw3E
        addw3dx0 = -M.dt*cE * adw3E
        addvdy0 = -M.dt*cE * adw3E
        #ddw3dx0 =  (3*dw30 - 4*dw30_ + dw30__)/(M.dx/2)
        adw30  +=  3*addw3dx0/(M.dx/2)
        adw30_  = -4*addw3dx0/(M.dx/2)
        adw30__ =    addw3dx0/(M.dx/2)
        # dw30__ = du0[:,-2] - (1/2)*np.sqrt(M.g/HeE**3)*(h0[:,-2]+h0[:,-3])/2 * dHeE +\
        #     np.sqrt(M.g/HeE)* (dh0[:,-2]+dh0[:,-3])/2
        adu_incr[:,-2] += adw30__
        adHeE += -(1/2)*np.sqrt(M.g/HeE**3)*(h0[:,-2]+h0[:,-3])/2 * adw30__
        adh_incr[:,-2] += np.sqrt(M.g/HeE)/2 * adw30__
        adh_incr[:,-3] += np.sqrt(M.g/HeE)/2 * adw30__
        # dw30_ = (du0[:,-1]+du0[:,-2])/2 - (1/2)*np.sqrt(M.g/HeE**3)*h0[:,-2] * dHeE +\
        #     np.sqrt(M.g/HeE)* dh0[:,-2]
        adu_incr[:,-1] += adw30_/2
        adu_incr[:,-2] += adw30_/2
        adHeE += -(1/2)*np.sqrt(M.g/HeE**3)* h0[:,-2] * adw30_
        adh_incr[:,-2] += np.sqrt(M.g/HeE)* adw30_
        # dw30 = du0[:,-1] - (1/2)*np.sqrt(M.g/HeE**3)*(h0[:,-1]+h0[:,-2])/2 * dHeE +\
        #     np.sqrt(M.g/HeE)* (dh0[:,-1]+dh0[:,-2])/2
        adu_incr[:,-1] += adw30
        adHeE += -(1/2)*np.sqrt(M.g/HeE**3)*(h0[:,-1]+h0[:,-2])/2 * adw30
        adh_incr[:,-2] += np.sqrt(M.g/HeE)* adw30/2
        adh_incr[:,-1] += np.sqrt(M.g/HeE)* adw30/2
    
    # Adj 2. w2
    if M.bc=='1d':
        # dw2E = dw20
        adw20 = +adw2E
    elif M.bc=='2d':
        # dw2E = dw20 - M.dt*M.g * ddhdy0 
        adw20 = +adw2E
        addhdy0 = -M.dt*M.g* adw2E
        # ddhdy0 = ((dh0[1:,-1]+dh0[1:,-2]-dh0[:-1,-1]-dh0[:-1,-2])/2)/M.dy
        adh_incr[1:,-1] += addhdy0/(2*M.dy)
        adh_incr[1:,-2] += addhdy0/(2*M.dy)
        adh_incr[:-1,-1] += -addhdy0/(2*M.dy)
        adh_incr[:-1,-2] += -addhdy0/(2*M.dy)
    # dw20 = (dv0[:,-1] + dv0[:,-2])/2
    adv_incr[:,-1] += adw20/2
    adv_incr[:,-2] += adw20/2
    
    # Adj 1. w1
    if  M.bc=='1d':   
        adw1_ext = +adw1E
    elif M.bc=='2d':
        # dw1E = dw10 + M.dt*(dcE* (dw1dx0 + dvdy0) + cE* (ddw1dx0 + ddvdy0))
        adw10 = +adw1E
        adcE += M.dt*(dw1dx0 + dvdy0) * adw1E
        addw1dx0 = M.dt*cE * adw1E
        addvdy0 += M.dt*cE * adw1E
        #ddvdy0[-1] = ddvdy0[-2]
        addvdy0[-2] += addvdy0[-1]
        #ddvdy0[0] = ddvdy0[1]
        addvdy0[1] += addvdy0[0]
        #ddvdy0[1:-1] = ((dv0[1:,-1] + dv0[1:,-2] - dv0[:-1,-1] - dv0[:-1,-2])/2)/M.dy
        adv_incr[1:,-1] += addvdy0[1:-1]/(2*M.dy)
        adv_incr[1:,-2] += addvdy0[1:-1]/(2*M.dy)
        adv_incr[:-1,-1] += -addvdy0[1:-1]/(2*M.dy)
        adv_incr[:-1,-2] += -addvdy0[1:-1]/(2*M.dy)
        #ddw1dx0 = (_dw10 - dw10_)/M.dx
        _adw10 = addw1dx0/M.dx
        adw10_ = -addw1dx0/M.dx
        #_dw10 = dw1_ext
        adw1_ext = +_adw10
        # dw10_= (du0[:,-1]+du0[:,-2])/2 + (1/2)*np.sqrt(M.g/HeE**3)*h0[:,-2] * dHeE -\
        #     np.sqrt(M.g/HeE)* dh0[:,-2]
        adu_incr[:,-1] += adw10_/2
        adu_incr[:,-2] += adw10_/2
        adHeE += (1/2)*np.sqrt(M.g/HeE**3)*h0[:,-2] * adw10_
        adh_incr[:,-2] += -np.sqrt(M.g/HeE)* adw10_
        # dw10 = du0[:,-1] + (1/2)*np.sqrt(M.g/HeE**3)*(h0[:,-1]+h0[:,-2])/2 * dHeE -\
        #     np.sqrt(M.g/HeE)* (dh0[:,-1]+dh0[:,-2])/2
        adu_incr[:,-1] += adw10
        adHeE += (1/2)*np.sqrt(M.g/HeE**3)*(h0[:,-1]+h0[:,-2])/2 * adw10
        adh_incr[:,-1] += -np.sqrt(M.g/HeE)/2* adw10
        adh_incr[:,-2] += -np.sqrt(M.g/HeE)/2* adw10
    for j,w in enumerate(M.omegas):
        k = np.sqrt((w**2-fE**2)/(M.g*HeE))
        #adh_ext = np.zeros(M.nx)
        #adv_ext = np.zeros(M.nx)
        for i,theta in enumerate(M.bc_theta):
            kx = -np.cos(theta) * k
            ky = np.sin(theta) * k
            kxy = kx*M.Xu[:,-1] + ky*M.Yu[:,-1]
            h_ext = hbcy[j,1,0,i]*np.cos(w*t-kxy) +\
                    hbcy[j,1,1,i]*np.sin(w*t-kxy)
            #dw1_ext += du_ext - np.sqrt(M.g/HeE) * dh_ext + \
            #     1/2*np.sqrt(M.g/HeE**3)*h_ext * dHeE 
            adh_ext = -np.sqrt(M.g/HeE) * adw1_ext
            adu_ext = +adw1_ext
            adHeE += 1/2*np.sqrt(M.g/HeE**3)*h_ext * adw1_ext
        #     du_ext = M.g/(w**2-fE**2)*(\
        #   dhbcy[j,1,0,i] * (w*kx*np.cos(w*t-kxy)+fE*ky*np.sin(w*t-kxy))\
        # +dkxy * hbcy[j,1,0,i]*(w*kx*np.sin(w*t-kxy)-fE*ky*np.cos(w*t-kxy))\
        # +dkx * hbcy[j,1,0,i]*w*np.cos(w*t-kxy)\
        # +dky * hbcy[j,1,0,i]*fE*np.sin(w*t-kxy)+\
        #   dhbcy[j,1,1,i] * (w*kx*np.sin(w*t-kxy)-fW*ky*np.cos(w*t-kxy))\
        # -dkxy * hbcy[j,1,1,i]*(w*kx*np.cos(w*t-kxy)+fW*ky*np.sin(w*t-kxy))\
        # +dkx * hbcy[j,1,1,i]*w*np.sin(w*t-kxy)\
        # -dky * hbcy[j,1,1,i]*fW*np.cos(w*t-kxy)) 
            adhbcy1d_incr[j,1,0,i] += M.g/(w**2-fE**2)*\
                (w*kx*np.cos(w*t-kxy)+fE*ky*np.sin(w*t-kxy)) * adu_ext
            adkxy = M.g/(w**2-fE**2)*\
                hbcy[j,1,0,i]*(w*kx*np.sin(w*t-kxy)-fE*ky*np.cos(w*t-kxy)) * adu_ext
            adkx = M.g/(w**2-fE**2)*\
                hbcy[j,1,0,i]*w*np.cos(w*t-kxy) * adu_ext
            adky = M.g/(w**2-fE**2)*\
                hbcy[j,1,0,i]*fE*np.sin(w*t-kxy) * adu_ext
            adhbcy1d_incr[j,1,1,i] += M.g/(w**2-fE**2)*\
                (w*kx*np.sin(w*t-kxy)-fE*ky*np.cos(w*t-kxy)) * adu_ext
            adkxy += -M.g/(w**2-fE**2)*\
                hbcy[j,1,1,i]*(w*kx*np.cos(w*t-kxy)+fE*ky*np.sin(w*t-kxy)) * adu_ext
            adkx += M.g/(w**2-fE**2)*\
                hbcy[j,1,1,i]*w*np.sin(w*t-kxy) * adu_ext
            adky += -M.g/(w**2-fE**2)*\
                hbcy[j,1,1,i]*fE*np.cos(w*t-kxy) * adu_ext
            # dh_ext = dhbcy[j,1,0,i] * np.cos(w*t-kxy) +\
            #         dkxy * hbcy[j,1,0,i]*np.sin(w*t-kxy) +\
            #          dhbcy[j,1,1,i] * np.sin(w*t-kxy) -\
            #         dkxy * hbcy[j,1,1,i]*np.cos(w*t-kxy)
            adhbcy1d_incr[j,1,0,i] += np.cos(w*t-kxy) * adh_ext
            adkxy += hbcy[j,1,0,i]*np.sin(w*t-kxy) * adh_ext
            adhbcy1d_incr[j,1,1,i] += np.sin(w*t-kxy) * adh_ext
            adkxy += -hbcy[j,1,1,i]*np.cos(w*t-kxy) * adh_ext
            #dkxy = -dHeE/HeE * kxy/2
            adHeE += -adkxy/HeE * kxy/2
            #dky = -dHeE/HeE * ky/2
            adHeE += -adky/HeE * ky/2
            #dkx = -dHeE/HeE * kx/2
            adHeE += -adkx/HeE * kx/2
            
    # Adj 0. Init boundary variables
    adHeE += cE/HeE/2 * adcE
    adHe2d_incr[:,-1] += adHeE/2
    adHe2d_incr[:,-2] += adHeE/2
    
    # Back to reduced shape
   # adhbcx_incr,adhbcy_incr = M.reduced_shape_hbc(
   #     t,adhbcx1d_incr,adhbcy1d_incr)
    
    return adu_incr,adv_incr,adh_incr,adHe2d_incr,adhbcx1d_incr,adhbcy1d_incr

def update_borders_adj(M,adu0,adv0,adh0):
    # Wouth
    aduS = np.zeros(M.Xu.shape[1])
    advS = np.zeros(M.Xv.shape[1])
    adhS = np.zeros(M.X.shape[1])
    aduS[1:-1] = 2*adu0[0,1:-1]
    adu0[1,1:-1] += -adu0[0,1:-1]
    adu0[0,1:-1] = 0
    advS[1:-1] = adv0[0,1:-1]
    adv0[0,1:-1] = 0
    adhS[1:-1] = 2*adh0[0,1:-1]
    adh0[1,1:-1] += -adh0[0,1:-1]
    adh0[0,1:-1] = 0
    
    # North
    aduN = np.zeros(M.Xu.shape[1])
    advN = np.zeros(M.Xv.shape[1])
    adhN = np.zeros(M.X.shape[1])
    aduN[1:-1] = 2*adu0[-1,1:-1]
    adu0[-2,1:-1] += -adu0[-1,1:-1]
    adu0[-1,1:-1] = 0
    advN[1:-1] = adv0[-1,1:-1]
    adv0[-1,1:-1] = 0
    adhN[1:-1] = 2*adh0[-1,1:-1]
    adh0[-2,1:-1] += -adh0[-1,1:-1]
    adh0[-1,1:-1] = 0
    
    # West
    aduW = np.zeros(M.Xu.shape[0])
    advW = np.zeros(M.Xv.shape[0])
    adhW = np.zeros(M.X.shape[0])
    aduW[1:-1] = adu0[1:-1,0]
    adu0[1:-1,0] = 0
    advW[1:-1] = 2*adv0[1:-1,0]
    adv0[1:-1,1] += -adv0[1:-1,0]
    adv0[1:-1,0] = 0
    adhW[1:-1] = 2*adh0[1:-1,0]
    adh0[1:-1,1] += -adh0[1:-1,0]
    adh0[1:-1,0] = 0
    
    # East
    aduE = np.zeros(M.Xu.shape[0])
    advE = np.zeros(M.Xv.shape[0])
    adhE = np.zeros(M.X.shape[0])
    aduE[1:-1] = adu0[1:-1,-1]
    adu0[1:-1,-1] = 0
    advE[1:-1] = 2*adv0[1:-1,-1]
    adv0[1:-1,-2] += -adv0[1:-1,-1]
    adv0[1:-1,-1] = 0
    adhE[1:-1] = 2*adh0[1:-1,-1]
    adh0[1:-1,-2] += -adh0[1:-1,-1]
    adh0[1:-1,-1] = 0
    
    # South-West
    aduS[0] = adu0[0,0]/2
    advS[0] = adv0[0,0]/2
    adhS[0] = adh0[0,0]/2
    aduW[0] = adu0[0,0]/2
    advW[0] = adv0[0,0]/2
    adhW[0] = adh0[0,0]/2
    adu0[0,0] = 0
    adv0[0,0] = 0
    adh0[0,0] = 0
    
    # South-East
    aduS[-1] = adu0[0,-1]/2
    advS[-1] = adv0[0,-1]/2
    adhS[-1] = adh0[0,-1]/2
    aduE[0] = adu0[0,-1]/2
    advE[0] = adv0[0,-1]/2
    adhE[0] = adh0[0,-1]/2
    adu0[0,-1] = 0
    adv0[0,-1] = 0
    adh0[0,-1] = 0
    
    # North-West
    aduN[0] = adu0[-1,0]/2
    advN[0] = adv0[-1,0]/2
    adhN[0] = adh0[-1,0]/2
    aduW[-1] = adu0[-1,0]/2
    advW[-1] = adv0[-1,0]/2
    adhW[-1] = adh0[-1,0]/2
    adu0[-1,0] = 0
    adv0[-1,0] = 0
    adh0[-1,0] = 0
    
    # North-East
    aduN[-1] = adu0[-1,-1]/2
    advN[-1] = adv0[-1,-1]/2
    adhN[-1] = adh0[-1,-1]/2
    aduE[-1] = adu0[-1,-1]/2
    advE[-1] = adv0[-1,-1]/2
    adhE[-1] = adh0[-1,-1]/2
    adu0[-1,-1] = 0
    adv0[-1,-1] = 0
    adh0[-1,-1] = 0
    
    return aduS,advS,adhS,aduN,advN,adhN,aduW,advW,adhW,aduE,advE,adhE