import numpy as np

###########################################################################
#                  Boundary conditions functions                          #
###########################################################################    

def init_bc_tgl(M,du,dv,dh,dHe,dhbcx,dhbcy,He,hbcx,hbcy,t0=0):
    
    duS = np.zeros(M.nx-1)
    dvS = np.zeros(M.nx)
    dhS = np.zeros(M.nx)
    duN = np.zeros(M.nx-1)
    dvN = np.zeros(M.nx)
    dhN = np.zeros(M.nx)
    duW = np.zeros(M.ny)
    dvW = np.zeros(M.ny-1)
    dhW = np.zeros(M.ny)
    duE = np.zeros(M.ny)
    dvE = np.zeros(M.ny-1)
    dhE = np.zeros(M.ny)
        
    for j,w in enumerate(M.omegas):
        for i,theta in enumerate(M.bc_theta):
            
            ###############################################################
            # South
            ###############################################################
            fS = (M.f[0,:]+M.f[1,:])/2
            HeS = (He[0,:]+He[1,:])/2
            dHeS = (dHe[0,:]+dHe[1,:])/2
            
            k = np.sqrt((w**2-fS**2)/(M.g*HeS))
            kx = np.sin(theta) * k
            dkx = -dHeS/HeS * kx/2
            ky = np.cos(theta) * k
            dky = -dHeS/HeS * ky/2
            
            kxy_h = kx*M.X[0,:] + ky*M.Y[0,:]
            kxy_v = kx*M.Xv[0,:] + ky*M.Yv[0,:]
            
            dkxy_h = -dHeS/HeS * kxy_h/2
            dkxy_v = -dHeS/HeS * kxy_v/2
            
            phase_h =  w*t0 - kxy_h
            phase_v =  w*t0 - kxy_v
            
            dhS += dhbcx[j,0,0,i] * np.cos(phase_h) +\
                dkxy_h * hbcx[j,0,0,i]*np.sin(phase_h) +\
                      dhbcx[j,0,1,i] * np.sin(phase_h) -\
                dkxy_h * hbcx[j,0,1,i]*np.cos(phase_h)
            
            dvS += M.g/(w**2-fS**2)*(\
          dhbcx[j,0,0,i] * (w*ky*np.cos(phase_v)-fS*kx*np.sin(phase_v))\
        +dkxy_v * hbcx[j,0,0,i]*(w*ky*np.sin(phase_v)+fS*kx*np.cos(phase_v))\
        +dky * hbcx[j,0,0,i]*w*np.cos(phase_v)\
        -dkx * hbcx[j,0,0,i]*fS*np.sin(phase_v) +\
          dhbcx[j,0,1,i] * (w*ky*np.sin(phase_v)+fS*kx*np.cos(phase_v))\
        -dkxy_v * hbcx[j,0,1,i]*(w*ky*np.cos(phase_v)-fS*kx*np.sin(phase_v))\
        +dky * hbcx[j,0,1,i]*w*np.sin(phase_v)\
        +dkx * hbcx[j,0,1,i]*fS*np.cos(phase_v)
        )  
            _duS = M.g/(w**2-fS**2)*(\
          dhbcx[j,0,0,i] * (w*kx*np.cos(phase_v)+fS*ky*np.sin(phase_v))\
        +dkxy_v * hbcx[j,0,0,i]*(w*kx*np.sin(phase_v)-fS*ky*np.cos(phase_v))\
        +dkx * hbcx[j,0,0,i]*w*np.cos(phase_v)\
        +dky * hbcx[j,0,0,i]*fS*np.sin(phase_v) +\
          dhbcx[j,0,1,i] * (w*kx*np.sin(phase_v)-fS*ky*np.cos(phase_v))\
        -dkxy_v * hbcx[j,0,1,i]*(w*kx*np.cos(phase_v)+fS*ky*np.sin(phase_v))\
        +dkx * hbcx[j,0,1,i]*w*np.sin(phase_v)\
        -dky * hbcx[j,0,1,i]*fS*np.cos(phase_v)) 
            duS += (_duS[1:] + _duS[:-1])/2
            
            
            ###############################################################
            # North
            ###############################################################
            fN = (M.f[-1,:]+M.f[-2,:])/2
            HeN = (He[-1,:]+He[-2,:])/2
            dHeN = (dHe[-1,:]+dHe[-2,:])/2
            
            k = np.sqrt((w**2-fN**2)/(M.g*HeN))
            kx = np.sin(theta) * k
            dkx = -dHeN/HeN * kx/2
            ky = -np.cos(theta) * k
            dky = -dHeN/HeN * ky/2
            
            kxy_h = kx*M.X[-1,:] + ky*M.Y[-1,:]
            kxy_v = kx*M.Xv[-1,:] + ky*M.Yv[-1,:]
            
            dkxy_h = -dHeN/HeN * kxy_h/2
            dkxy_v = -dHeN/HeN * kxy_v/2
            
            phase_h =  w*t0 - kxy_h
            phase_v =  w*t0 - kxy_v
            
            dhN += dhbcx[j,1,0,i] * np.cos(phase_h) +\
                dkxy_h * hbcx[j,1,0,i]*np.sin(phase_h) +\
                      dhbcx[j,1,1,i] * np.sin(phase_h) -\
                dkxy_h * hbcx[j,1,1,i]*np.cos(phase_h)
                      
                       
            dvN += M.g/(w**2-fN**2)*(\
          dhbcx[j,1,0,i] * (w*ky*np.cos(phase_v)-fN*kx*np.sin(phase_v))\
        +dkxy_v * hbcx[j,1,0,i]*(w*ky*np.sin(phase_v)+fN*kx*np.cos(phase_v))\
        +dky * hbcx[j,1,0,i]*w*np.cos(phase_v)\
        -dkx * hbcx[j,1,0,i]*fN*np.sin(phase_v) +\
          dhbcx[j,1,1,i] * (w*ky*np.sin(phase_v)+fN*kx*np.cos(phase_v))\
        -dkxy_v * hbcx[j,1,1,i]*(w*ky*np.cos(phase_v)-fN*kx*np.sin(phase_v))\
        +dky * hbcx[j,1,1,i]*w*np.sin(phase_v)\
        +dkx * hbcx[j,1,1,i]*fN*np.cos(phase_v)
        ) 
            _duN = M.g/(w**2-fN**2)*(\
          dhbcx[j,1,0,i] * (w*kx*np.cos(phase_v)+fN*ky*np.sin(phase_v))\
        +dkxy_v * hbcx[j,1,0,i]*(w*kx*np.sin(phase_v)-fN*ky*np.cos(phase_v))\
        +dkx * hbcx[j,1,0,i]*w*np.cos(phase_v)\
        +dky * hbcx[j,1,0,i]*fN*np.sin(phase_v) +\
          dhbcx[j,1,1,i] * (w*kx*np.sin(phase_v)-fN*ky*np.cos(phase_v))\
        -dkxy_v * hbcx[j,1,1,i]*(w*kx*np.cos(phase_v)+fN*ky*np.sin(phase_v))\
        +dkx * hbcx[j,1,1,i]*w*np.sin(phase_v)\
        -dky * hbcx[j,1,1,i]*fN*np.cos(phase_v)
        ) 
            duN += (_duN[1:] + _duN[:-1])/2
            
            ###############################################################
            # West
            ###############################################################
            fW = (M.f[:,0]+M.f[:,1])/2
            HeW = (He[:,0]+He[:,1])/2
            dHeW = (dHe[:,0]+dHe[:,1])/2
            
            k = np.sqrt((w**2-fW**2)/(M.g*HeW))
            kx = np.cos(theta) * k
            dkx = -dHeW/HeW * kx/2
            ky = np.sin(theta) * k
            dky = -dHeW/HeW * ky/2
            
            kxy_h = kx*M.X[:,0] + ky*M.Y[:,0]
            kxy_u = kx*M.Xu[:,0] + ky*M.Yu[:,0]
            
            dkxy_h = -dHeW/HeW * kxy_h/2
            dkxy_u = -dHeW/HeW * kxy_u/2
            
            phase_h =  w*t0 - kxy_h
            phase_u =  w*t0 - kxy_u
            
            
            dhW += dhbcy[j,0,0,i] * np.cos(phase_h) +\
                    dkxy_h * hbcy[j,0,0,i]*np.sin(phase_h) +\
                      dhbcy[j,0,1,i] * np.sin(phase_h) -\
                    dkxy_h * hbcy[j,0,1,i]*np.cos(phase_h)
                       
            duW += M.g/(w**2-fW**2)*(\
          dhbcy[j,0,0,i] * (w*kx*np.cos(phase_u)+fW*ky*np.sin(phase_u))\
        +dkxy_u * hbcy[j,0,0,i]*(w*kx*np.sin(phase_u)-fW*ky*np.cos(phase_u))\
        +dkx * hbcy[j,0,0,i]*w*np.cos(phase_u)\
        +dky * hbcy[j,0,0,i]*fW*np.sin(phase_u) +\
          dhbcy[j,0,1,i] * (w*kx*np.sin(phase_u)-fW*ky*np.cos(phase_u))\
        -dkxy_u * hbcy[j,0,1,i]*(w*kx*np.cos(phase_u)+fW*ky*np.sin(phase_u))\
        +dkx * hbcy[j,0,1,i]*w*np.sin(phase_u)\
        -dky * hbcy[j,0,1,i]*fW*np.cos(phase_u)
        ) 
            _dvW = M.g/(w**2-fW**2)*(\
          dhbcy[j,0,0,i] * (w*ky*np.cos(phase_u)-fW*kx*np.sin(phase_u))\
        +dkxy_u * hbcy[j,0,0,i]*(w*ky*np.sin(phase_u)+fW*kx*np.cos(phase_u))\
        +dky * hbcy[j,0,0,i]*w*np.cos(phase_u)\
        -dkx * hbcy[j,0,0,i]*fW*np.sin(phase_u) +\
          dhbcy[j,0,1,i] * (w*ky*np.sin(phase_u)+fW*kx*np.cos(phase_u))\
        -dkxy_u * hbcy[j,0,1,i]*(w*ky*np.cos(phase_u)-fW*kx*np.sin(phase_u))\
        +dky * hbcy[j,0,1,i]*w*np.sin(phase_u)\
        +dkx * hbcy[j,0,1,i]*fW*np.cos(phase_u)
        )  
            dvW += (_dvW[1:] + _dvW[:-1])/2
            
            ###############################################################
            # East
            ###############################################################
            fE = (M.f[:,-1]+M.f[:,-2])/2
            HeE = (He[:,-1]+He[:,-2])/2
            dHeE = (dHe[:,-1]+dHe[:,-2])/2
            
            k = np.sqrt((w**2-fE**2)/(M.g*HeE))
            kx = -np.cos(theta) * k
            dkx = -dHeE/HeE * kx/2
            ky = np.sin(theta) * k
            dky = -dHeE/HeE * ky/2
            
            kxy_h = kx*M.X[:,-1] + ky*M.Y[:,-1]
            kxy_u = kx*M.Xu[:,-1] + ky*M.Yu[:,-1]
            
            dkxy_h = -dHeE/HeE * kxy_h/2
            dkxy_u = -dHeE/HeE * kxy_u/2
            
            phase_h =  w*t0 - kxy_h
            phase_u =  w*t0 - kxy_u
            
            
            dhE += dhbcy[j,1,0,i] * np.cos(phase_h) +\
                    dkxy_h * hbcy[j,1,0,i]*np.sin(phase_h) +\
                      dhbcy[j,1,1,i] * np.sin(phase_h) -\
                    dkxy_h * hbcy[j,1,1,i]*np.cos(phase_h)
                       
            duE += M.g/(w**2-fE**2)*(\
          dhbcy[j,1,0,i] * (w*kx*np.cos(phase_u)+fE*ky*np.sin(phase_u))\
        +dkxy_u * hbcy[j,1,0,i]*(w*kx*np.sin(phase_u)-fE*ky*np.cos(phase_u))\
        +dkx * hbcy[j,1,0,i]*w*np.cos(phase_u)\
        +dky * hbcy[j,1,0,i]*fE*np.sin(phase_u) +\
          dhbcy[j,1,1,i] * (w*kx*np.sin(phase_u)-fE*ky*np.cos(phase_u))\
        -dkxy_u * hbcy[j,1,1,i]*(w*kx*np.cos(phase_u)+fE*ky*np.sin(phase_u))\
        +dkx * hbcy[j,1,1,i]*w*np.sin(phase_u)\
        -dky * hbcy[j,1,1,i]*fE*np.cos(phase_u)
        ) 
            _dvE = M.g/(w**2-fE**2)*(\
          dhbcy[j,1,0,i] * (w*ky*np.cos(phase_u)-fE*kx*np.sin(phase_u))\
        +dkxy_u * hbcy[j,1,0,i]*(w*ky*np.sin(phase_u)+fE*kx*np.cos(phase_u))\
        +dky * hbcy[j,1,0,i]*w*np.cos(phase_u)\
        -dkx * hbcy[j,1,0,i]*fE*np.sin(phase_u) +\
          dhbcy[j,1,1,i] * (w*ky*np.sin(phase_u)+fE*kx*np.cos(phase_u))\
        -dkxy_u * hbcy[j,1,1,i]*(w*ky*np.cos(phase_u)-fE*kx*np.sin(phase_u))\
        +dky * hbcy[j,1,1,i]*w*np.sin(phase_u)\
        +dkx * hbcy[j,1,1,i]*fE*np.cos(phase_u)
        )  
            dvE += (_dvE[1:] + _dvE[:-1])/2
            
    
    update_borders_tgl(M,du,dv,dh,
                       duS,dvS,dhS,duN,dvN,dhN,duW,dvW,dhW,duE,dvE,dhE)
    
    
    
    

def obcs_tgl(M,t,du0,dv0,dh0,dHe,dhbcx,dhbcy,u0,v0,h0,He,hbcx,hbcy):

    if M.bc=='1d':
        t += M.dt
    
    #######################################################################
    # South
    #######################################################################
    fS = (M.f[0,:]+M.f[1,:])/2
    HeS = (He[0,:]+He[1,:])/2
    dHeS = (dHe[0,:]+dHe[1,:])/2
    cS = np.sqrt(M.g*HeS)
    if M.bc=='1d':
        cS *= M.dt/(M.Y[1,:]-M.Y[0,:])
    dcS = cS/HeS/2 * dHeS

    # 1. w1
    w1_ext = np.zeros(M.nx)
    dw1_ext = np.zeros(M.nx)
    for j,w in enumerate(M.omegas):
        k = np.sqrt((w**2-fS**2)/(M.g*HeS))
        for i,theta in enumerate(M.bc_theta):
            kx = np.sin(theta) * k
            dkx = -dHeS/HeS * kx/2
            ky = np.cos(theta) * k
            dky = -dHeS/HeS * ky/2
            kxy = kx*M.Xv[0,:] + ky*M.Yv[0,:]
            dkxy = -dHeS/HeS * kxy/2
            h_ext = hbcx[j,0,0,i]* np.cos(w*t-kxy)  +\
                    hbcx[j,0,1,i]* np.sin(w*t-kxy) 
            dh_ext = dhbcx[j,0,0,i] * np.cos(w*t-kxy) +\
                dkxy * hbcx[j,0,0,i]*np.sin(w*t-kxy) +\
                      dhbcx[j,0,1,i] * np.sin(w*t-kxy) -\
                dkxy * hbcx[j,0,1,i]*np.cos(w*t-kxy)
            v_ext = M.g/(w**2-fS**2)*(\
                hbcx[j,0,0,i]* (w*ky*np.cos(w*t-kxy) \
                            - fS*kx*np.sin(w*t-kxy)
                                ) +\
                hbcx[j,0,1,i]* (w*ky*np.sin(w*t-kxy) \
                            + fS*kx*np.cos(w*t-kxy)
                                )
                    )
            dv_ext = M.g/(w**2-fS**2)*(\
          dhbcx[j,0,0,i] * (w*ky*np.cos(w*t-kxy)-fS*kx*np.sin(w*t-kxy))\
        +dkxy * hbcx[j,0,0,i]*(w*ky*np.sin(w*t-kxy)+fS*kx*np.cos(w*t-kxy))\
        +dky * hbcx[j,0,0,i]*w*np.cos(w*t-kxy)\
        -dkx * hbcx[j,0,0,i]*fS*np.sin(w*t-kxy) +\
          dhbcx[j,0,1,i] * (w*ky*np.sin(w*t-kxy)+fS*kx*np.cos(w*t-kxy))\
        -dkxy * hbcx[j,0,1,i]*(w*ky*np.cos(w*t-kxy)-fS*kx*np.sin(w*t-kxy))\
        +dky * hbcx[j,0,1,i]*w*np.sin(w*t-kxy)\
        +dkx * hbcx[j,0,1,i]*fS*np.cos(w*t-kxy))
                
            w1_ext += v_ext + np.sqrt(M.g/HeS) * h_ext                                   
            dw1_ext += dv_ext + np.sqrt(M.g/HeS) * dh_ext - \
                1/2*np.sqrt(M.g/HeS**3)*h_ext * dHeS 
    if M.bc=='1d':
        w1S = w1_ext
        dw1S = dw1_ext
    elif M.bc=='2d':
        # dw1dy0
        w10  = v0[0,:] + np.sqrt(M.g/HeS)* (h0[0,:]+h0[1,:])/2
        dw10 = dv0[0,:] - (1/2)*np.sqrt(M.g/HeS**3)*(h0[0,:]+h0[1,:])/2 * dHeS +\
            np.sqrt(M.g/HeS)* (dh0[0,:]+dh0[1,:])/2
        w10_  = (v0[0,:]+v0[1,:])/2 + np.sqrt(M.g/HeS)* h0[1,:]
        dw10_ = (dv0[0,:]+dv0[1,:])/2 - (1/2)*np.sqrt(M.g/HeS**3)*h0[1,:] * dHeS +\
            np.sqrt(M.g/HeS)* dh0[1,:]
        _w10 = w1_ext
        _dw10 = dw1_ext
        dw1dy0 = (w10_ - _w10)/M.dy
        ddw1dy0 = (dw10_ - _dw10)/M.dy
        # dudx0
        dudx0 = np.zeros(M.nx)
        ddudx0 = np.zeros(M.nx)
        dudx0[1:-1] = ((u0[0,1:] + u0[1,1:] - u0[0,:-1] - u0[1,:-1])/2)/M.dx
        ddudx0[1:-1] = ((du0[0,1:] + du0[1,1:] - du0[0,:-1] - du0[1,:-1])/2)/M.dx
        dudx0[0] = dudx0[1]
        ddudx0[0] = ddudx0[1]
        dudx0[-1] = dudx0[-2]
        ddudx0[-1] = ddudx0[-2]
        # w1S
        w1S = w10 - M.dt*cS* (dw1dy0 + dudx0)
        dw1S = dw10 - M.dt*(dcS* (dw1dy0 + dudx0) + cS* (ddw1dy0 + ddudx0))
        
    # 2. w2
    dw20 = (du0[0,:] + du0[1,:])/2
    if M.bc=='1d':
        dw2S = dw20
    elif M.bc=='2d':
        # dhdx0
        ddhdx0 = ((dh0[0,1:]+dh0[1,1:]-dh0[0,:-1]-dh0[1,:-1])/2)/M.dx
        dw2S = dw20 - M.dt*M.g* ddhdx0 
            
    # 3. w3
    if M.bc=='1d':
        _vS = (1-3/2*cS)* v0[0,:] + cS/2* (4*v0[1,:] - v0[2,:])
        _dvS = -3/2*v0[0,:]* dcS + (4*v0[1,:] - v0[2,:])/2 * dcS + \
            (1-3/2*cS)* dv0[0,:] + cS/2* (4*dv0[1,:] - dv0[2,:])
        _hS = (1/2+cS)* h0[1,:] + (1/2-cS)* h0[0,:]
        _dhS = h0[1,:] * dcS - h0[0,:] * dcS + (1/2+cS)* dh0[1,:] +\
            (1/2-cS)* dh0[0,:]
        w3S = _vS - np.sqrt(M.g/HeS) * _hS
        dw3S = _dvS - np.sqrt(M.g/HeS) * _dhS +\
            1/2*np.sqrt(M.g/HeS**3)*_hS * dHeS 
    elif M.bc=='2d':
        w30  = v0[0,:] - np.sqrt(M.g/HeS)* (h0[0,:]+h0[1,:])/2
        dw30 = dv0[0,:] + (1/2)*np.sqrt(M.g/HeS**3)*(h0[0,:]+h0[1,:])/2 * dHeS -\
            np.sqrt(M.g/HeS)* (dh0[0,:]+dh0[1,:])/2
        w30_   = (v0[0,:]+v0[1,:])/2  - np.sqrt(M.g/HeS)* h0[1,:]
        dw30_  = (dv0[0,:]+dv0[1,:])/2 + (1/2)*np.sqrt(M.g/HeS**3)* h0[1,:] * dHeS -\
            np.sqrt(M.g/HeS)* dh0[1,:]
        w30__  = v0[1,:] - np.sqrt(M.g/HeS)* (h0[1,:]+h0[2,:])/2
        dw30__ = dv0[1,:] + (1/2)*np.sqrt(M.g/HeS**3)*(h0[1,:]+h0[2,:])/2 * dHeS -\
            np.sqrt(M.g/HeS)* (dh0[1,:]+dh0[2,:])/2
        dw3dy0 =  -(3*w30 - 4*w30_ + w30__)/(M.dy/2)
        ddw3dy0 =  -(3*dw30 - 4*dw30_ + dw30__)/(M.dy/2)
        w3S = w30 + M.dt*cS* (dw3dy0 + dudx0) 
        dw3S = dw30 + M.dt*(dcS* (dw3dy0 + dudx0) + cS* (ddw3dy0 + ddudx0))
    
    # 4. Values on BC
    duS = dw2S
    dvS = (dw1S + dw3S)/2 
    dhS = 1/2 * (np.sqrt(HeS/M.g) * (dw1S - dw3S) +\
                  1/(2*np.sqrt(HeS*M.g))*(w1S-w3S) * dHeS)
    
    #######################################################################
    # North
    #######################################################################
    fN = (M.f[-1,:]+M.f[-2,:])/2
    HeN = (He[-1,:]+He[-2,:])/2
    dHeN = (dHe[-1,:]+dHe[-2,:])/2
    cN = np.sqrt(M.g*HeN)
    if M.bc=='1d':
        cN *= M.dt/(M.Y[-1,:]-M.Y[-2,:])
    dcN = cN/HeN/2 * dHeN
    
    # 1. w1
    w1_ext = np.zeros(M.nx)
    dw1_ext = np.zeros(M.nx)
    for j,w in enumerate(M.omegas):
        k = np.sqrt((w**2-fN**2)/(M.g*HeN))
        for i,theta in enumerate(M.bc_theta):
            kx = np.sin(theta) * k
            dkx = -dHeN/HeN * kx/2
            ky = -np.cos(theta) * k
            dky = -dHeN/HeN * ky/2
            kxy = kx*M.Xv[-1,:] + ky*M.Yv[-1,:]
            dkxy = -dHeN/HeN * kxy/2
            h_ext = hbcx[j,1,0,i]* np.cos(w*t-kxy)+\
                    hbcx[j,1,1,i]* np.sin(w*t-kxy) 
            dh_ext = dhbcx[j,1,0,i] * np.cos(w*t-kxy) +\
                    dkxy * hbcx[j,1,0,i]*np.sin(w*t-kxy) +\
                      dhbcx[j,1,1,i] * np.sin(w*t-kxy) -\
                    dkxy * hbcx[j,1,1,i]*np.cos(w*t-kxy)
            v_ext = M.g/(w**2-fN**2)*(\
                hbcx[j,1,0,i]* (w*ky*np.cos(w*t-kxy) \
                            - fN*kx*np.sin(w*t-kxy)
                                ) +\
                hbcx[j,1,1,i]* (w*ky*np.sin(w*t-kxy) \
                            + fN*kx*np.cos(w*t-kxy)
                                )
                    )
            dv_ext = M.g/(w**2-fN**2)*(\
          dhbcx[j,1,0,i] * (w*ky*np.cos(w*t-kxy)-fN*kx*np.sin(w*t-kxy))\
        +dkxy * hbcx[j,1,0,i]*(w*ky*np.sin(w*t-kxy)+fN*kx*np.cos(w*t-kxy))\
        +dky * hbcx[j,1,0,i]*w*np.cos(w*t-kxy)\
        -dkx * hbcx[j,1,0,i]*fN*np.sin(w*t-kxy) +\
          dhbcx[j,1,1,i] * (w*ky*np.sin(w*t-kxy)+fN*kx*np.cos(w*t-kxy))\
        -dkxy * hbcx[j,1,1,i]*(w*ky*np.cos(w*t-kxy)-fN*kx*np.sin(w*t-kxy))\
        +dky * hbcx[j,1,1,i]*w*np.sin(w*t-kxy)\
        +dkx * hbcx[j,1,1,i]*fN*np.cos(w*t-kxy))
            w1_ext += v_ext - np.sqrt(M.g/HeN) * h_ext                                   
            dw1_ext += dv_ext - np.sqrt(M.g/HeN) * dh_ext + \
                1/2*np.sqrt(M.g/HeN**3)*h_ext * dHeN
    if M.bc=='1d':
        w1N = w1_ext
        dw1N = dw1_ext
    elif M.bc=='2d':
        # dw1dy0
        w10  = v0[-1,:] - np.sqrt(M.g/HeN)* (h0[-1,:]+h0[-2,:])/2
        dw10 = dv0[-1,:] + (1/2)*np.sqrt(M.g/HeN**3)*(h0[-1,:]+h0[-2,:])/2 * dHeN -\
            np.sqrt(M.g/HeN)* (dh0[-1,:]+dh0[-2,:])/2
        w10_  = (v0[-1,:]+v0[-2,:])/2 - np.sqrt(M.g/HeN)* h0[-2,:]
        dw10_ = (dv0[-1,:]+dv0[-2,:])/2 + (1/2)*np.sqrt(M.g/HeN**3)*h0[-2,:] * dHeN -\
            np.sqrt(M.g/HeN)* dh0[-2,:]
        _w10 = w1_ext
        _dw10 = dw1_ext
        dw1dy0 = (_w10 - w10_)/M.dy
        ddw1dy0 = (_dw10 - dw10_)/M.dy
        # dudx0
        dudx0 = np.zeros(M.nx)
        ddudx0 = np.zeros(M.nx)
        dudx0[1:-1] = ((u0[-1,1:] + u0[-2,1:] - u0[-1,:-1] - u0[-2,:-1])/2)/M.dx
        ddudx0[1:-1] = ((du0[-1,1:] + du0[-2,1:] - du0[-1,:-1] - du0[-2,:-1])/2)/M.dx
        dudx0[0] = dudx0[1]
        ddudx0[0] = ddudx0[1]
        dudx0[-1] = dudx0[-2]
        ddudx0[-1] = ddudx0[-2]
        # w1N
        w1N = w10 + M.dt*cN* (dw1dy0 + dudx0)
        dw1N = dw10 + M.dt*(dcN* (dw1dy0 + dudx0) + cN* (ddw1dy0 + ddudx0))
        
    # 2. w2
    dw20 = (du0[-1,:] + du0[-2,:])/2
    if M.bc=='1d':   
        dw2N = dw20
    elif M.bc=='2d':
        ddhdx0 = ((dh0[-1,1:]+dh0[-2,1:]-dh0[-1,:-1]-dh0[-2,:-1])/2)/M.dx
        dw2N = dw20 - M.dt*M.g*ddhdx0 
            
    # 3. w3
    if M.bc=='1d':   
        _vN = (1-3/2*cN)* v0[-1,:] + cN/2* (4*v0[-2,:] - v0[-3,:])
        _dvN = -3/2*v0[-1,:]* dcN + (4*v0[-2,:] - v0[-3,:])/2 * dcN + \
            (1-3/2*cN)* dv0[-1,:] + cN/2* (4*dv0[-2,:] - dv0[-3,:])
        _hN = (1/2+cN)* h0[-2,:] + (1/2-cN)* h0[-1,:]
        _dhN = h0[-2,:] * dcN - h0[-1,:] * dcN + (1/2+cN)* dh0[-2,:] +\
            (1/2-cN)* dh0[-1,:]
        w3N = _vN + np.sqrt(M.g/HeN) * _hN
        dw3N = _dvN + np.sqrt(M.g/HeN) * _dhN -\
            1/2*np.sqrt(M.g/HeN**3)*_hN  * dHeN 
    elif M.bc=='2d':
        w30  = v0[-1,:] + np.sqrt(M.g/HeN)* (h0[-1,:]+h0[-2,:])/2
        dw30 = dv0[-1,:] - (1/2)*np.sqrt(M.g/HeN**3)*(h0[-1,:]+h0[-2,:])/2 * dHeN +\
            np.sqrt(M.g/HeN)* (dh0[-1,:]+dh0[-2,:])/2
        w30_   = (v0[-1,:]+v0[-2,:])/2 + np.sqrt(M.g/HeN)* h0[-2,:]
        dw30_  = (dv0[-1,:]+dv0[-2,:])/2 - (1/2)*np.sqrt(M.g/HeN**3)* h0[-2,:] * dHeN +\
            np.sqrt(M.g/HeN)* dh0[-2,:]
        w30__  = v0[-2,:] + np.sqrt(M.g/HeN)* (h0[-2,:]+h0[-3,:])/2
        dw30__ = dv0[-2,:] - (1/2)*np.sqrt(M.g/HeN**3)*(h0[-2,:]+h0[-3,:])/2 * dHeN +\
            np.sqrt(M.g/HeN)* (dh0[-2,:]+dh0[-3,:])/2
        dw3dy0 =  (3*w30 - 4*w30_ + w30__)/(M.dy/2)
        ddw3dy0 =  (3*dw30 - 4*dw30_ + dw30__)/(M.dy/2)
        w3N = w30 - M.dt*cN* (dw3dy0 + dudx0) 
        dw3N = dw30 - M.dt*(dcN* (dw3dy0 + dudx0) + cN* (ddw3dy0 + ddudx0))
    # 4. Values on BC
    duN = dw2N
    dvN = (dw1N + dw3N)/2 
    dhN = 1/2 * (np.sqrt(HeN/M.g) * (dw3N - dw1N) +\
                  1/(2*np.sqrt(HeN*M.g))*(w3N-w1N) * dHeN)
    
    #######################################################################
    # West
    #######################################################################
    fW = (M.f[:,0]+M.f[:,1])/2
    HeW = (He[:,0]+He[:,1])/2
    dHeW = (dHe[:,0]+dHe[:,1])/2
    cW = np.sqrt(M.g*HeW)
    if M.bc=='1d':
        cW *= M.dt/(M.X[:,1]-M.X[:,0])
    dcW = cW/HeW/2 * dHeW
    
    # 1. w1
    w1_ext = np.zeros(M.ny)
    dw1_ext = np.zeros(M.ny)
    for j,w in enumerate(M.omegas):
        k = np.sqrt((w**2-fW**2)/(M.g*HeW))
        for i,theta in enumerate(M.bc_theta):
            kx = np.cos(theta)* k
            dkx = -dHeW/HeW * kx/2
            ky = np.sin(theta)* k
            dky = -dHeW/HeW * ky/2
            kxy = kx*M.Xu[:,0] + ky*M.Yu[:,0]
            dkxy = -dHeW/HeW * kxy/2
            h_ext = hbcy[j,0,0,i]*np.cos(w*t-kxy) +\
                    hbcy[j,0,1,i]*np.sin(w*t-kxy)
            dh_ext = dhbcy[j,0,0,i] * np.cos(w*t-kxy) +\
                    dkxy * hbcy[j,0,0,i]*np.sin(w*t-kxy) +\
                      dhbcy[j,0,1,i] * np.sin(w*t-kxy) -\
                    dkxy * hbcy[j,0,1,i]*np.cos(w*t-kxy)
            u_ext = M.g/(w**2-fW**2)*(\
                hbcy[j,0,0,i]*(w*kx*np.cos(w*t-kxy) \
                          + fW*ky*np.sin(w*t-kxy)
                              ) +\
                hbcy[j,0,1,i]*(w*kx*np.sin(w*t-kxy) \
                          - fW*ky*np.cos(w*t-kxy)
                              )
                    )
            du_ext = M.g/(w**2-fW**2)*(\
          dhbcy[j,0,0,i] * (w*kx*np.cos(w*t-kxy)+fW*ky*np.sin(w*t-kxy))\
        +dkxy * hbcy[j,0,0,i]*(w*kx*np.sin(w*t-kxy)-fW*ky*np.cos(w*t-kxy))\
        +dkx * hbcy[j,0,0,i]*w*np.cos(w*t-kxy)\
        +dky * hbcy[j,0,0,i]*fW*np.sin(w*t-kxy) +\
          dhbcy[j,0,1,i] * (w*kx*np.sin(w*t-kxy)-fW*ky*np.cos(w*t-kxy))\
        -dkxy * hbcy[j,0,1,i]*(w*kx*np.cos(w*t-kxy)+fW*ky*np.sin(w*t-kxy))\
        +dkx * hbcy[j,0,1,i]*w*np.sin(w*t-kxy)\
        -dky * hbcy[j,0,1,i]*fW*np.cos(w*t-kxy)) 
            w1_ext += u_ext + np.sqrt(M.g/HeW) * h_ext                                   
            dw1_ext += du_ext + np.sqrt(M.g/HeW) * dh_ext - \
                1/2*np.sqrt(M.g/HeW**3)*h_ext * dHeW 
    if M.bc=='1d':   
        w1W = w1_ext
        dw1W = dw1_ext
    elif M.bc=='2d':
        w10  = u0[:,0] + np.sqrt(M.g/HeW)* (h0[:,0]+h0[:,1])/2
        dw10  = du0[:,0] - (1/2)*np.sqrt(M.g/HeW**3)*(h0[:,0]+h0[:,1])/2 * dHeW +\
            np.sqrt(M.g/HeW)* (dh0[:,0]+dh0[:,1])/2 
        w10_ = (u0[:,0]+u0[:,1])/2 + np.sqrt(M.g/HeW)* h0[:,1]
        dw10_ = (du0[:,0]+du0[:,1])/2 - (1/2)*np.sqrt(M.g/HeW**3)*h0[:,1] * dHeW +\
            np.sqrt(M.g/HeW)* dh0[:,1]
        _w10 = w1_ext
        _dw10 = dw1_ext
        dw1dx0 = (w10_ - _w10)/M.dx
        ddw1dx0 = (dw10_ - _dw10)/M.dx
        dvdy0 = np.zeros(M.ny)
        ddvdy0 = np.zeros(M.ny)
        dvdy0[1:-1] = ((v0[1:,0] + v0[1:,1] - v0[:-1,0] - v0[:-1,1])/2)/M.dy
        ddvdy0[1:-1] = ((dv0[1:,0] + dv0[1:,1] - dv0[:-1,0] - dv0[:-1,1])/2)/M.dy
        dvdy0[0] = dvdy0[1]
        ddvdy0[0] = ddvdy0[1]
        dvdy0[-1] = dvdy0[-2]
        ddvdy0[-1] = ddvdy0[-2]
        w1W = w10 - M.dt*cW* (dw1dx0 + dvdy0) 
        dw1W = dw10 - M.dt*(dcW* (dw1dx0 + dvdy0) + cW* (ddw1dx0 + ddvdy0))
    
    # 2. w2
    dw20 = (dv0[:,0] + dv0[:,1])/2
    if M.bc=='1d':   
        dw2W = dw20
    elif M.bc=='2d':
        ddhdy0 = ((dh0[1:,0]+dh0[1:,1]-dh0[:-1,0]-dh0[:-1,1])/2)/M.dy
        dw2W = dw20 - M.dt*M.g * ddhdy0 
            
    # 3. w3
    if M.bc=='1d':   
        _uW = (1-3/2*cW)* u0[:,0] + cW/2* (4*u0[:,1] - u0[:,2])
        _duW = -3/2*u0[:,0]* dcW + (4*u0[:,1] - u0[:,2])/2 * dcW + \
            (1-3/2*cW)* du0[:,0] + cW/2* (4*du0[:,1] - du0[:,2])
        _hW = (1/2+cW)* h0[:,1] + (1/2-cW)* h0[:,0]
        _dhW= h0[:,1] * dcW - h0[:,0] * dcW + (1/2+cW)* dh0[:,1] +\
            (1/2-cW)* dh0[:,0]
        w3W = _uW - np.sqrt(M.g/HeW) * _hW
        dw3W = _duW - np.sqrt(M.g/HeW) * _dhW +\
            1/2*np.sqrt(M.g/HeW**3)*_hW * dHeW 
    elif M.bc=='2d':
        w30   = u0[:,0] - np.sqrt(M.g/HeW)* (h0[:,0]+h0[:,1])/2
        dw30  = du0[:,0] + (1/2)*np.sqrt(M.g/HeW**3)*(h0[:,0]+h0[:,1])/2 * dHeW -\
            np.sqrt(M.g/HeW)* (dh0[:,0]+dh0[:,1])/2
        w30_  = (u0[:,0]+u0[:,1])/2 - np.sqrt(M.g/HeW)* h0[:,1]
        dw30_ = (du0[:,0]+du0[:,1])/2 + (1/2)*np.sqrt(M.g/HeW**3)*h0[:,1] * dHeW -\
            np.sqrt(M.g/HeW)* dh0[:,1]
        w30__ = u0[:,1] - np.sqrt(M.g/HeW)* (h0[:,1]+h0[:,2])/2
        dw30__= du0[:,1] + (1/2)*np.sqrt(M.g/HeW**3)*(h0[:,1]+h0[:,2])/2 * dHeW -\
            np.sqrt(M.g/HeW)* (dh0[:,1]+dh0[:,2])/2
        dw3dx0 = -(3*w30 - 4*w30_ + w30__)/(M.dx/2)
        ddw3dx0 = -(3*dw30 - 4*dw30_ + dw30__)/(M.dx/2)
        w3W = w30 + M.dt*cW* (dw3dx0 + dvdy0)
        dw3W = dw30 + M.dt*(dcW* (dw3dx0 + dvdy0) + cW* (ddw3dx0 + ddvdy0))
    
    # 4. Values on BC
    duW = (dw1W + dw3W)/2 
    dvW = dw2W
    dhW = 1/2 * (np.sqrt(HeW/M.g) * (dw1W - dw3W) +\
                  1/(2*np.sqrt(HeW*M.g))*(w1W-w3W) * dHeW)
    
    #######################################################################
    # East
    #######################################################################
    fE = (M.f[:,-1]+M.f[:,-2])/2
    HeE = (He[:,-1]+He[:,-2])/2
    dHeE = (dHe[:,-1]+dHe[:,-2])/2
    cE = np.sqrt(M.g*HeE)
    if M.bc=='1d':
        cE *= M.dt/(M.X[:,-1]-M.X[:,-2])
    dcE = cE/HeE/2 * dHeE
    
    # 1. w1
    w1_ext = np.zeros(M.ny)
    dw1_ext = np.zeros(M.ny)
    for j,w in enumerate(M.omegas):
        k = np.sqrt((w**2-fE**2)/(M.g*HeE))
        for i,theta in enumerate(M.bc_theta):
            kx = -np.cos(theta)* k
            dkx = -dHeE/HeE * kx/2
            ky = np.sin(theta)* k
            dky = -dHeE/HeE * ky/2
            kxy = kx*M.Xu[:,-1] + ky*M.Yu[:,-1]
            dkxy = -dHeE/HeE * kxy/2
            h_ext = hbcy[j,1,0,i]*np.cos(w*t-kxy) +\
                    hbcy[j,1,1,i]*np.sin(w*t-kxy)
            dh_ext = dhbcy[j,1,0,i] * np.cos(w*t-kxy) +\
                    dkxy * hbcy[j,1,0,i]*np.sin(w*t-kxy) +\
                      dhbcy[j,1,1,i] * np.sin(w*t-kxy) -\
                    dkxy * hbcy[j,1,1,i]*np.cos(w*t-kxy)
            u_ext = M.g/(w**2-fE**2)*(\
                hbcy[j,1,0,i]* (w*kx*np.cos(w*t-kxy) \
                            + fE*ky*np.sin(w*t-kxy)
                                ) +\
                hbcy[j,1,1,i]*(w*kx*np.sin(w*t-kxy) \
                          - fE*ky*np.cos(w*t-kxy)
                              )
                    )
            du_ext = M.g/(w**2-fE**2)*(\
          dhbcy[j,1,0,i] * (w*kx*np.cos(w*t-kxy)+fE*ky*np.sin(w*t-kxy))\
        +dkxy * hbcy[j,1,0,i]*(w*kx*np.sin(w*t-kxy)-fE*ky*np.cos(w*t-kxy))\
        +dkx * hbcy[j,1,0,i]*w*np.cos(w*t-kxy)\
        +dky * hbcy[j,1,0,i]*fE*np.sin(w*t-kxy)+\
          dhbcy[j,1,1,i] * (w*kx*np.sin(w*t-kxy)-fE*ky*np.cos(w*t-kxy))\
        -dkxy * hbcy[j,1,1,i]*(w*kx*np.cos(w*t-kxy)+fE*ky*np.sin(w*t-kxy))\
        +dkx * hbcy[j,1,1,i]*w*np.sin(w*t-kxy)\
        -dky * hbcy[j,1,1,i]*fE*np.cos(w*t-kxy)) 
            w1_ext += u_ext - np.sqrt(M.g/HeE) * h_ext                                   
            dw1_ext += du_ext - np.sqrt(M.g/HeE) * dh_ext + \
                1/2*np.sqrt(M.g/HeE**3)*h_ext * dHeE 
    if M.bc=='1d':   
        w1E = w1_ext
        dw1E = dw1_ext
    elif M.bc=='2d':
        w10  = u0[:,-1] - np.sqrt(M.g/HeE)* (h0[:,-1]+h0[:,-2])/2
        dw10 = du0[:,-1] + (1/2)*np.sqrt(M.g/HeE**3)*(h0[:,-1]+h0[:,-2])/2 * dHeE -\
            np.sqrt(M.g/HeE)* (dh0[:,-1]+dh0[:,-2])/2
        w10_ = (u0[:,-1]+u0[:,-2])/2 - np.sqrt(M.g/HeE)* h0[:,-2]
        dw10_= (du0[:,-1]+du0[:,-2])/2 + (1/2)*np.sqrt(M.g/HeE**3)*h0[:,-2] * dHeE -\
            np.sqrt(M.g/HeE)* dh0[:,-2]
        _w10  = w1_ext
        _dw10 = dw1_ext
        dw1dx0 = (_w10 - w10_)/M.dx
        ddw1dx0 = (_dw10 - dw10_)/M.dx
        dvdy0 = np.zeros(M.ny)
        ddvdy0 = np.zeros(M.ny)
        dvdy0[1:-1] = ((v0[1:,-1] + v0[1:,-2] - v0[:-1,-1] - v0[:-1,-2])/2)/M.dy
        ddvdy0[1:-1] = ((dv0[1:,-1] + dv0[1:,-2] - dv0[:-1,-1] - dv0[:-1,-2])/2)/M.dy
        dvdy0[0] = dvdy0[1]
        ddvdy0[0] = ddvdy0[1]
        dvdy0[-1] = dvdy0[-2]
        ddvdy0[-1] = ddvdy0[-2]
        w1E = w10 + M.dt*cE* (dw1dx0 + dvdy0) 
        dw1E = dw10 + M.dt*(dcE* (dw1dx0 + dvdy0) + cE* (ddw1dx0 + ddvdy0))
        
    # 2. w2
    dw20 = (dv0[:,-1] + dv0[:,-2])/2
    if  M.bc=='1d':   
        dw2E = dw20
    elif M.bc=='2d':
        dw20 = (dv0[:,-1] + dv0[:,-2])/2
        ddhdy0 = ((dh0[1:,-1]+dh0[1:,-2]-dh0[:-1,-1]-dh0[:-1,-2])/2)/M.dy
        dw2E = dw20 - M.dt*M.g * ddhdy0 
            
    # 3. w3
    if M.bc=='1d':   
        _uE = (1-3/2*cE)* u0[:,-1] + cE/2* (4*u0[:,-2] - u0[:,-3])
        _duE = -3/2*u0[:,-1]* dcE + (4*u0[:,-2] - u0[:,-3])/2 * dcE + \
            (1-3/2*cE)* du0[:,-1] + cE/2* (4*du0[:,-2] - du0[:,-3])
        _hE = (1/2+cE)* h0[:,-2] + (1/2-cE)* h0[:,-1]
        _dhE= h0[:,-2] * dcE - h0[:,-1] * dcE + (1/2+cE)* dh0[:,-2] +\
            (1/2-cE)* dh0[:,-1]
        w3E = _uE + np.sqrt(M.g/HeE) * _hE
        dw3E = _duE + np.sqrt(M.g/HeE) * _dhE -\
            1/2*np.sqrt(M.g/HeE**3)*_hE * dHeE 
    elif M.bc=='2d':
        w30  = u0[:,-1] + np.sqrt(M.g/HeE)* (h0[:,-1]+h0[:,-2])/2
        dw30 = du0[:,-1] - (1/2)*np.sqrt(M.g/HeE**3)*(h0[:,-1]+h0[:,-2])/2 * dHeE +\
            np.sqrt(M.g/HeE)* (dh0[:,-1]+dh0[:,-2])/2
        w30_  = (u0[:,-1]+u0[:,-2])/2 + np.sqrt(M.g/HeE)* h0[:,-2]
        dw30_ = (du0[:,-1]+du0[:,-2])/2 - (1/2)*np.sqrt(M.g/HeE**3)*h0[:,-2] * dHeE +\
            np.sqrt(M.g/HeE)* dh0[:,-2]
        w30__ = u0[:,-2] + np.sqrt(M.g/HeE)* (h0[:,-2]+h0[:,-3])/2
        dw30__ = du0[:,-2] - (1/2)*np.sqrt(M.g/HeE**3)*(h0[:,-2]+h0[:,-3])/2 * dHeE +\
            np.sqrt(M.g/HeE)* (dh0[:,-2]+dh0[:,-3])/2
        dw3dx0 =  (3*w30 - 4*w30_ + w30__)/(M.dx/2)
        ddw3dx0 =  (3*dw30 - 4*dw30_ + dw30__)/(M.dx/2)
        w3E  = w30 - M.dt*cE* (dw3dx0 + dvdy0) 
        dw3E = dw30 - M.dt*(dcE* (dw3dx0 + dvdy0) + cE* (ddw3dx0 + ddvdy0))
    # 4. Values on BC
    duE = (dw1E + dw3E)/2 
    dvE = dw2E
    dhE = 1/2 * (np.sqrt(HeE/M.g) * (dw3E - dw1E) +\
                  1/(2*np.sqrt(HeE*M.g))*(w3E-w1E) * dHeE)
    
        
    return duS,dvS,dhS,duN,dvN,dhN,duW,dvW,dhW,duE,dvE,dhE


def update_borders_tgl(M,du,dv,dh,
                       duS,dvS,dhS,duN,dvN,dhN,duW,dvW,dhW,duE,dvE,dhE):
    # South
    du[0,1:-1] = 2* duS[1:-1] - du[1,1:-1]
    dv[0,1:-1] = dvS[1:-1]
    dh[0,1:-1] = 2* dhS[1:-1] - dh[1,1:-1]
    # North
    du[-1,1:-1] = 2* duN[1:-1] - du[-2,1:-1]
    dv[-1,1:-1] = dvN[1:-1]
    dh[-1,1:-1] = 2* dhN[1:-1] - dh[-2,1:-1]
    # West
    du[1:-1,0] = duW[1:-1]
    dv[1:-1,0] = 2* dvW[1:-1] - dv[1:-1,1]
    dh[1:-1,0] = 2* dhW[1:-1] - dh[1:-1,1]
    # East
    du[1:-1,-1] = duE[1:-1]
    dv[1:-1,-1] = 2* dvE[1:-1] - dv[1:-1,-2]
    dh[1:-1,-1] = 2* dhE[1:-1] - dh[1:-1,-2]
    # South-West
    du[0,0] = (duS[0] + duW[0])/2
    dv[0,0] = (dvS[0] + dvW[0])/2
    dh[0,0] = (dhS[0] + dhW[0])/2
    # South-East
    du[0,-1] = (duS[-1] + duE[0])/2
    dv[0,-1] = (dvS[-1] + dvE[0])/2
    dh[0,-1] = (dhS[-1] + dhE[0])/2
    # North-West
    du[-1,0] = (duN[0] + duW[-1])/2
    dv[-1,0] = (dvN[0] + dvW[-1])/2
    dh[-1,0] = (dhN[0] + dhW[-1])/2
    # North-East
    du[-1,-1] = (duN[-1] + duE[-1])/2
    dv[-1,-1] = (dvN[-1] + dvE[-1])/2
    dh[-1,-1] = (dhN[-1] + dhE[-1])/2
    