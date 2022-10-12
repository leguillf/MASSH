import numpy as np

###########################################################################
#                  Boundary conditions functions                          #
###########################################################################    
    
def init_bc(M,u,v,h,He2d,hbcx,hbcy,t0=0):
    
    uS = np.zeros(M.nx-1)
    vS = np.zeros(M.nx)
    hS = np.zeros(M.nx)
    uN = np.zeros(M.nx-1)
    vN = np.zeros(M.nx)
    hN = np.zeros(M.nx)
    uW = np.zeros(M.ny)
    vW = np.zeros(M.ny-1)
    hW = np.zeros(M.ny)
    uE = np.zeros(M.ny)
    vE = np.zeros(M.ny-1)
    hE = np.zeros(M.ny)

    for j,w in enumerate(M.omegas):
        for i,theta in enumerate(M.bc_theta):
            
            ###############################################################
            # South
            ###############################################################
            fS = (M.f[0,:]+M.f[1,:])/2
            HeS = (He2d[0,:]+He2d[1,:])/2
            
            k = np.sqrt((w**2-fS**2)/(M.g*HeS))
            kx = np.sin(theta) * k
            ky = np.cos(theta) * k
            
            phase_h =  w*t0 - kx*M.X[0,:] - ky*M.Y[0,:]
            phase_v =  w*t0 - kx*M.Xv[0,:] - ky*M.Yv[0,:]
            
            hS += hbcx[j,0,0,i]* np.cos(phase_h)  +\
                  hbcx[j,0,1,i]* np.sin(phase_h)
                       
            vS += M.g/(w**2-fS**2)*(\
                hbcx[j,0,0,i]* (w*ky*np.cos(phase_v) \
                            -fS*kx*np.sin(phase_v)
                                ) +\
                hbcx[j,0,1,i]* (w*ky*np.sin(phase_v) \
                            +fS*kx*np.cos(phase_v)
                                )
                    )
            _uS = M.g/(w**2-fS**2)*(\
                hbcx[j,0,0,i]* (w*kx*np.cos(phase_v) \
                            +fS*ky*np.sin(phase_v)
                                ) +\
                hbcx[j,0,1,i]* (w*kx*np.sin(phase_v) \
                            -fS*ky*np.cos(phase_v)
                                )
                    )
            uS += (_uS[1:] + _uS[:-1])/2
            
            
            ###############################################################
            # North
            ###############################################################
            fN = (M.f[-1,:]+M.f[-2,:])/2
            HeN = (He2d[-1,:]+He2d[-2,:])/2
            
            k = np.sqrt((w**2-fN**2)/(M.g*HeN))
            kx = np.sin(theta) * k
            ky = -np.cos(theta) * k
            
            phase_h =  w*t0 - kx*M.X[-1,:] - ky*M.Y[-1,:]
            phase_v =  w*t0 - kx*M.Xv[-1,:] - ky*M.Yv[-1,:]
            
            hN += hbcx[j,1,0,i]* np.cos(phase_h)  +\
                  hbcx[j,1,1,i]* np.sin(phase_h)
                       
            vN += M.g/(w**2-fN**2)*(\
                hbcx[j,1,0,i]* (w*ky*np.cos(phase_v) \
                            -fN*kx*np.sin(phase_v)
                                ) +\
                hbcx[j,1,1,i]* (w*ky*np.sin(phase_v) \
                            +fN*kx*np.cos(phase_v)
                                )
                    )
            _uN = M.g/(w**2-fN**2)*(\
                hbcx[j,1,0,i]* (w*kx*np.cos(phase_v) \
                            +fN*ky*np.sin(phase_v)
                                ) +\
                hbcx[j,1,1,i]* (w*kx*np.sin(phase_v) \
                            -fN*ky*np.cos(phase_v)
                                )
                    )
            uN += (_uN[1:] + _uN[:-1])/2
            
            ###############################################################
            # West
            ###############################################################
            fW = (M.f[:,0]+M.f[:,1])/2
            HeW = (He2d[:,0]+He2d[:,1])/2
            
            k = np.sqrt((w**2-fW**2)/(M.g*HeW))
            kx = np.cos(theta) * k
            ky = np.sin(theta) * k
            
            phase_h =  w*t0 - kx*M.X[:,0] - ky*M.Y[:,0]
            phase_u =  w*t0 - kx*M.Xu[:,0] - ky*M.Yu[:,0]
            
            hW += hbcy[j,0,0,i]* np.cos(phase_h)  +\
                  hbcy[j,0,1,i]* np.sin(phase_h)
                       
            uW += M.g/(w**2-fW**2)*(\
                hbcy[j,0,0,i]* (w*kx*np.cos(phase_u) \
                            +fW*ky*np.sin(phase_u)
                                ) +\
                hbcy[j,0,1,i]* (w*kx*np.sin(phase_u) \
                            -fW*ky*np.cos(phase_u)
                                )
                    )
            _vW = M.g/(w**2-fW**2)*(\
                hbcy[j,0,0,i]* (w*ky*np.cos(phase_u) \
                            -fW*kx*np.sin(phase_u)
                                ) +\
                hbcy[j,0,1,i]* (w*ky*np.sin(phase_u) \
                            +fW*kx*np.cos(phase_u)
                                )
                    )
            vW += (_vW[1:] + _vW[:-1])/2
            
            ###############################################################
            # East
            ###############################################################
            fE = (M.f[:,-1]+M.f[:,-2])/2
            HeE = (He2d[:,-1]+He2d[:,-2])/2
            
            k = np.sqrt((w**2-fE**2)/(M.g*HeE))
            kx = -np.cos(theta) * k
            ky = np.sin(theta) * k
            
            phase_h =  w*t0 - kx*M.X[:,-1] - ky*M.Y[:,-1]
            phase_u =  w*t0 - kx*M.Xu[:,-1] - ky*M.Yu[:,-1]
            
            hE += hbcy[j,1,0,i]* np.cos(phase_h)  +\
                        hbcy[j,1,1,i]* np.sin(phase_h)
                       
            uE += M.g/(w**2-fE**2)*(\
                hbcy[j,1,0,i]* (w*kx*np.cos(phase_u) \
                            +fE*ky*np.sin(phase_u)
                                ) +\
                hbcy[j,1,1,i]* (w*kx*np.sin(phase_u) \
                            -fE*ky*np.cos(phase_u)
                                )
                    )
            _vE = M.g/(w**2-fE**2)*(\
                hbcy[j,1,0,i]* (w*ky*np.cos(phase_u) \
                            -fE*kx*np.sin(phase_u)
                                ) +\
                hbcy[j,1,1,i]* (w*ky*np.sin(phase_u) \
                            +fE*kx*np.cos(phase_u)
                                )
                    )
            vE += (_vE[1:] + _vE[:-1])/2
            
    update_borders(M,u,v,h,
                       uS,vS,hS,uN,vN,hN,uW,vW,hW,uE,vE,hE)
            

    
    
def obcs(M,t,u,v,h,u0,v0,h0,He,hbcx,hbcy):
    
    if M.bc=='1d':
        t += M.dt
            
    #######################################################################
    # South
    #######################################################################
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
            v_ext = M.g/(w**2-fS**2)*( \
                hbcx[j,0,0,i]* (w*ky*np.cos(w*t-kxy) \
                            - fS*kx*np.sin(w*t-kxy)
                                ) +\
                hbcx[j,0,1,i]* (w*ky*np.sin(w*t-kxy) \
                            + fS*kx*np.cos(w*t-kxy)
                                )
                    )
        
            _w1_ext = v_ext + np.sqrt(M.g/HeS) * h_ext
            w1_ext += _w1_ext
    
    if M.bc=='1d':
        w1S = w1_ext
    elif M.bc=='2d':
        # dw1dy0
        w10  = v0[0,:] + np.sqrt(M.g/HeS)* (h0[0,:]+h0[1,:])/2
        w10_ = (v0[0,:]+v0[1,:])/2 + np.sqrt(M.g/HeS)* h0[1,:]
        _w10 = w1_ext
        dw1dy0 = (w10_ - _w10)/M.dy
        # dudx0
        dudx0 = np.zeros(M.nx)
        dudx0[1:-1] = ((u0[0,1:] + u0[1,1:] - u0[0,:-1] - u0[1,:-1])/2)/M.dx
        dudx0[0] = dudx0[1]
        dudx0[-1] = dudx0[-2]
        # w1S
        w1S = w10 - M.dt*cS* (dw1dy0 + dudx0)
    
    # 2. w2
    w20 = (u0[0,:] + u0[1,:])/2
    if M.bc=='1d':
        w2S = w20
    elif M.bc=='2d':
        dhdx0 = ((h0[0,1:]+h0[1,1:]-h0[0,:-1]-h0[1,:-1])/2)/M.dx
        w2S = w20 - M.dt*M.g* dhdx0 
            
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

    # 4. Values on BC
    uS = w2S
    vS = (w1S + w3S)/2 
    hS = np.sqrt(HeS/M.g) *(w1S - w3S)/2
    
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
            _w1_ext = v_ext - np.sqrt(M.g/HeN) * h_ext
            w1_ext += _w1_ext
    
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
        
    # 2. w2
    w20 = (u0[-1,:] + u0[-2,:])/2
    if M.bc=='1d':   
        w2N = w20
    elif M.bc=='2d':
        dhdx0 = ((h0[-1,1:]+h0[-2,1:]-h0[-1,:-1]-h0[-2,:-1])/2)/M.dx
        w2N = w20 - M.dt*M.g*dhdx0 
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
    
    # 4. Values on BC
    uN = w2N
    vN = (w1N + w3N)/2 
    hN = np.sqrt(HeN/M.g) *(w3N - w1N)/2
    
    #######################################################################
    # West
    #######################################################################
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
            kx = np.cos(theta)* k
            ky = np.sin(theta)* k
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
            _w1_ext = u_ext + np.sqrt(M.g/HeW) * h_ext
            w1_ext += _w1_ext
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
        
    # 2. w2
    w20 = (v0[:,0] + v0[:,1])/2
    if M.bc=='1d':   
        w2W = w20
    elif M.bc=='2d':
        dhdy0 = ((h0[1:,0]+h0[1:,1]-h0[:-1,0]-h0[:-1,1])/2)/M.dy
        w2W = w20 - M.dt*M.g * dhdy0 
            
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
        
    # 4. Values on BC
    uW = (w1W + w3W)/2 
    vW = w2W
    hW = np.sqrt(HeW/M.g)*(w1W - w3W)/2
    
    #######################################################################
    # East
    #######################################################################
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
            kx = -np.cos(theta)* k
            ky = np.sin(theta)* k
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
    # 2. w2
    w20 = (v0[:,-1] + v0[:,-2])/2
    if  M.bc=='1d':   
        w2E = w20
    elif M.bc=='2d':
        w20 = (v0[:,-1] + v0[:,-2])/2
        dhdy0 = ((h0[1:,-1]+h0[1:,-2]-h0[:-1,-1]-h0[:-1,-2])/2)/M.dy
        w2E = w20 - M.dt*M.g * dhdy0 
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
        
    # 4. Values on BC
    uE = (w1E + w3E)/2 
    vE = w2E
    hE = np.sqrt(HeE/M.g)*(w3E - w1E)/2
    
    
    update_borders(M,u,v,h,
                       uS,vS,hS,uN,vN,hN,uW,vW,hW,uE,vE,hE)
    
    

def update_borders(M,u,v,h,
                       uS,vS,hS,uN,vN,hN,uW,vW,hW,uE,vE,hE):
    #######################################################################
    # Update border pixels 
    #######################################################################
    # South
    u[0,1:-1] = 2* uS[1:-1] - u[1,1:-1]
    v[0,1:-1] = vS[1:-1]
    h[0,1:-1] = 2* hS[1:-1] - h[1,1:-1]
    
    # North
    u[-1,1:-1] = 2* uN[1:-1] - u[-2,1:-1]
    v[-1,1:-1] = vN[1:-1]
    h[-1,1:-1] = 2* hN[1:-1] - h[-2,1:-1]
    # West
    u[1:-1,0] = uW[1:-1]
    v[1:-1,0] = 2* vW[1:-1] - v[1:-1,1]
    h[1:-1,0] = 2* hW[1:-1] - h[1:-1,1]
    # East
    u[1:-1,-1] = uE[1:-1]
    v[1:-1,-1] = 2* vE[1:-1] - v[1:-1,-2]
    h[1:-1,-1] = 2* hE[1:-1] - h[1:-1,-2]
    # South-West
    u[0,0] = (uS[0] + uW[0])/2
    v[0,0] = (vS[0] + vW[0])/2
    h[0,0] = (hS[0] + hW[0])/2
    # South-East
    u[0,-1] = (uS[-1] + uE[0])/2
    v[0,-1] = (vS[-1] + vE[0])/2
    h[0,-1] = (hS[-1] + hE[0])/2
    # North-West
    u[-1,0] = (uN[0] + uW[-1])/2
    v[-1,0] = (vN[0] + vW[-1])/2
    h[-1,0] = (hN[0] + hW[-1])/2
    # North-East
    u[-1,-1] = (uN[-1] + uE[-1])/2
    v[-1,-1] = (vN[-1] + vE[-1])/2
    h[-1,-1] = (hN[-1] + hE[-1])/2
    
    return u,v,h
