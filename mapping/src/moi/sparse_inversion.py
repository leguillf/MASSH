import sys,os,shutil
import numpy
from numpy import pi, zeros
from scipy import interpolate
import os.path
import scipy.io
import netCDF4 as nc
import pdb
import matplotlib.pylab as plt
import time
import shutil
import scipy
import copy

import logging

from scipy.sparse import csc_matrix, coo_matrix, csr_matrix
import scipy.sparse

#from mpi4py import MPI




def build_gmatrix(comp, obs, obs_data,  params_algo, comm=None, rank=0, size=1):


    nko = len(obs)
    nkc = len(comp)

    nobs_loc=0
    for ko in range(nko):
        nobs_loc += obs_data[ko][3]['nobs']


    cumsize = numpy.zeros((1), dtype=params_algo['int_type'])
    G = csc_matrix((numpy.empty((0),dtype=params_algo['float_type']), numpy.empty((0),dtype=params_algo['int_type']), cumsize),   shape=(nobs_loc, 0), dtype=params_algo['float_type']) 
    for kc in range(len(comp)):
        cumsize = numpy.zeros((comp[kc].nwave + 1), dtype=params_algo['int_type'])
        tmp_mat = csr_matrix(csc_matrix((numpy.empty((0),dtype=params_algo['float_type']), numpy.empty((0),dtype=params_algo['int_type']), cumsize),   shape=(0, comp[kc].nwave), dtype=params_algo['float_type']) )

        for ko in range(len(obs)):
            if ((ko in comp[kc].obs_datasets)&(obs_data[ko][3]['nobs']>0)):
                
                result = comp[kc].operg(coords=obs_data[ko][2],coords_name=obs_data[ko][3], nature=obs[ko].nature, compute_g=True, 
                                        iwave0=0, iwave1=comp[kc].nwave, obs_name=obs[ko].name, gsize_max=params_algo['gsize_max'], 
                                        int_type = params_algo['int_type'], float_type = params_algo['float_type'], label='from proc: '+str(rank))
                cumsize = numpy.empty((comp[kc].nwave + 1), dtype=params_algo['int_type'])
                cumsize[0] = 0
                cumsize[1:] = numpy.cumsum(result[0])
                tmp_mat=csc_matrix(scipy.sparse.vstack((tmp_mat, csr_matrix(csc_matrix((result[2], result[1], cumsize),   shape=(obs_data[ko][3]['nobs'], comp[kc].nwave), dtype=params_algo['float_type']))) ) )

            else:
                # csc_matrix a mettre aussi ici comme pour le cas transpose
                cumsize = numpy.zeros((comp[kc].nwave + 1), dtype=params_algo['int_type'])
                tmp_mat = csc_matrix(scipy.sparse.vstack((tmp_mat , csr_matrix(csc_matrix((numpy.empty((0),dtype=params_algo['float_type']), numpy.empty((0),dtype=params_algo['int_type']), cumsize),   shape=(obs_data[ko][3]['nobs'], comp[kc].nwave), dtype=params_algo['float_type']) )) ) )

        G = scipy.sparse.hstack(( G , csc_matrix(tmp_mat) )) 

    logging.info('SIZE OF G on proc %s : %s and %s', rank,  G.nnz, G.shape)

    cumsize = G.indptr
    G = csr_matrix(G)

    if params_algo['store_gtranspose']:

        #cumsize = G.indptr
        cumsize_glo = comm.reduce(numpy.array(cumsize, dtype=params_algo['int_type']), op=MPI.SUM, root=0)
        comm.barrier()

        if rank==0:
            logging.info('')
            logging.info('START TRANSPOSING G')
            logging.info('')  
    
            val_splits = numpy.array(numpy.linspace(0,cumsize_glo[-1], size+1), dtype=params_algo['int_type'])

            split_waves = numpy.searchsorted(cumsize_glo, val_splits)
            split_waves[0]=0
            split_waves[-1]=len(cumsize_glo)-1
            logging.info('split_waves : %s', split_waves)

        else:
            split_waves = None    
        split_waves = comm.bcast(split_waves, root=0)
        comm.barrier()


        nobs_tot = sum(comm.allgather(nobs_loc))
        split_obs = numpy.zeros((size+1), dtype=params_algo['int_type'])
        split_obs[1:] = numpy.cumsum(comm.allgather(nobs_loc))

        split_obs_ko=[None]*len(obs)
        obs_data_glo = [None]*len(obs)
        for ko in range(len(obs)):
            obs_data_glo[ko]=[None]*4
            obs_data_glo[ko][3]=copy.copy(obs_data[ko][3])
            obs_data_glo[ko][3]['nobs'] = comm.allreduce(obs_data[ko][3]['nobs'], op=MPI.SUM)

            
            split_obs_ko[ko] = numpy.zeros((size+1), dtype=params_algo['int_type'])
            list_nobs = comm.allgather(obs_data[ko][3]['nobs'])
            split_obs_ko[ko][1:]=numpy.cumsum(list_nobs)

            obs_data_glo[ko][2]=[None]*len(obs_data[ko][2])
            for kk in range(len(obs_data[ko][2])):
                comm.barrier()
                # obs_data_glo[ko][2][kk]=numpy.empty((obs_data_glo[ko][3]['nobs']),dtype='f4')
                # sendcounts = numpy.array(comm.gather(obs_data[ko][3]['nobs'], 0))
                # comm.Gatherv(sendbuf=obs_data[ko][2][kk], recvbuf=(obs_data_glo[ko][2][kk], sendcounts), root=0)
                # obs_data_glo[ko][2][kk][:] = comm.bcast(obs_data_glo[ko][2][kk], root=0) 
                obs_data_glo[ko][2][kk]=numpy.empty((obs_data_glo[ko][3]['nobs']),dtype='f8')
                sendcounts_obs = numpy.array(comm.allgather(len(obs_data[ko][2][kk])))
                offsets_obs = numpy.zeros(size)
                offsets_obs[1:]=numpy.cumsum(sendcounts_obs)[:-1]
                comm.Allgatherv(numpy.array(obs_data[ko][2][kk], dtype='f8'), [obs_data_glo[ko][2][kk],sendcounts_obs,offsets_obs,MPI.DOUBLE])
                logging.debug('from proc : %s , nobs: %s',rank, obs_data[ko][3]['nobs'] )
                #if ((rank==0)&(kk==0)): pdb.set_trace()



        iwave=0
        iwave0_glo=numpy.empty((len(comp)), dtype=params_algo['int_type'])
        for kc in range(len(comp)):
            iwave0_glo[kc]=iwave
            iwave += comp[kc].nwave
        comm.barrier()
        if rank==0: logging.debug('iwave0_glo : %s', iwave0_glo)
        
        cumsize = numpy.zeros((1), dtype=params_algo['int_type'])
        nwaves_loc = split_waves[rank+1] - split_waves[rank]

        G2 = csc_matrix((numpy.empty((0),dtype=params_algo['float_type']), numpy.empty((0),dtype=params_algo['int_type']), cumsize),   shape=(nobs_tot, 0), dtype=params_algo['float_type']) 
        for kc in range(len(comp)):

            if ((split_waves[rank] < iwave0_glo[kc]+comp[kc].nwave)&(split_waves[rank+1] >= iwave0_glo[kc])):
                iwave0=numpy.maximum(0,split_waves[rank]-iwave0_glo[kc])
                iwave1=numpy.minimum(comp[kc].nwave,split_waves[rank+1]-iwave0_glo[kc])
                logging.debug('On proc : %s , comp: %s, iwave0: %s iwave1: %s', rank, kc, iwave0, iwave1)

                cumsize = numpy.zeros((iwave1-iwave0 + 1), dtype=params_algo['int_type'])
                tmp_mat = csr_matrix(csc_matrix((numpy.empty((0),dtype=params_algo['float_type']), numpy.empty((0),dtype=params_algo['int_type']), cumsize),   shape=(0, iwave1-iwave0), dtype=params_algo['float_type']) )

                for ko in range(len(obs)):
                    if ((ko in comp[kc].obs_datasets)&(obs_data_glo[ko][3]['nobs']>0)):
                        # if rank==0: logging.info('OBS DATA: %s', obs_data_glo[ko][2])
                        result = comp[kc].operg(coords=obs_data_glo[ko][2],coords_name=obs_data_glo[ko][3], nature=obs[ko].nature, compute_g=True, 
                                                iwave0=iwave0, iwave1=iwave1, obs_name=obs[ko].name, gsize_max=params_algo['gsize_max'], 
                                                int_type = params_algo['int_type'], float_type = params_algo['float_type'], label='from proc: '+str(rank))
                        cumsize = numpy.empty((iwave1-iwave0 + 1), dtype=params_algo['int_type'])
                        cumsize[0] = 0
                        cumsize[1:] = numpy.cumsum(result[0])
                        #if rank==0: pdb.set_trace()
                        tmp_mat=csc_matrix(scipy.sparse.vstack((tmp_mat, csr_matrix(csc_matrix((result[2], result[1], cumsize),   shape=(obs_data_glo[ko][3]['nobs'], iwave1-iwave0), dtype=params_algo['float_type']))) ) )
                        #tmp_mat=scipy.sparse.vstack((tmp_mat, csc_matrix((result[2], result[1], cumsize),   shape=(iend-istart, comp[kc].nwave), dtype=params_algo['float_type'])))
                    else:
                        cumsize = numpy.zeros((iwave1-iwave0 + 1), dtype=params_algo['int_type'])
                        tmp_mat=csc_matrix(scipy.sparse.vstack((tmp_mat, csr_matrix(csc_matrix((numpy.empty((0),dtype=params_algo['float_type']), numpy.empty((0),dtype=params_algo['int_type']), cumsize),   shape=(obs_data_glo[ko][3]['nobs'], iwave1-iwave0), dtype=params_algo['float_type']) )) ) )
                
                G2 = scipy.sparse.hstack(( G2 , csc_matrix(tmp_mat) )) 

        logging.debug('On proc : %s , start transposing G2 block', rank)
        G2 = csr_matrix(G2.transpose())
        logging.debug('On proc : %s , end transposing G2 block', rank)

        if rank==0:
            iobs0_ko=numpy.zeros((len(obs)),dtype=params_algo['int_type'])
            i0=0
            for ko in range(len(obs)):
               iobs0_ko[ko] = i0 
               i0 += obs_data_glo[ko][3]['nobs']

            indsort = numpy.empty((nobs_tot),dtype=params_algo['int_type'] )
            i0=0 
            

            for ipr in range(size):
                for ko in range(len(obs)):
                    nind = split_obs_ko[ko][ipr+1] - split_obs_ko[ko][ipr]
                    iref = iobs0_ko[ko] + split_obs_ko[ko][ipr] #split_obs[ipr] + 
                    indsort[iref:iref+nind] = i0 + numpy.arange(0,nind)
                    #indsort[split_obs_ko[ko][ipr]:split_obs_ko[ko][ipr]+nind] = i0+numpy.arange(0,nind)
                    i0 += nind
        else: indsort = None
        indsort = comm.bcast(indsort, root=0)
        #if rank==0: pdb.set_trace()
        comm.barrier()
        logging.debug('On proc : %s , sort indices', rank)
        G2.indices = indsort[G2.indices]

        logging.info('SIZE OF G2 on proc %s : %s and %s', rank,  G2.nnz, G2.shape)
        comm.barrier()

        return [G, G2, split_waves]
    else:

        return [G, None, None]



    

def solve_pcg(Gdata, comp_Qinv, obs_invnoise2_loc, obs_val_loc, params_algo,  rank=0, comm=None):

    nobs_loc = len(obs_invnoise2_loc)
    

    G = Gdata[0]
    Gt = Gdata[1]
    if Gt is not None:
        split_waves = Gdata[2]
        iw0 = split_waves[rank]
        iw1 = split_waves[rank+1]

        size = comm.Get_size()   
        nobs_tot = sum(comm.allgather(nobs_loc))
    
        obs_invnoise2=numpy.empty((nobs_tot),dtype='f8')
        sendcounts_obs = numpy.array(comm.allgather(len(obs_invnoise2_loc)))
        offsets_obs = numpy.zeros(size)
        offsets_obs[1:]=numpy.cumsum(sendcounts_obs)[:-1]
        comm.Allgatherv(obs_invnoise2_loc, [obs_invnoise2,sendcounts_obs,offsets_obs,MPI.DOUBLE])

        obs_val=numpy.empty((nobs_tot),dtype='f8')
        comm.Allgatherv(obs_val_loc, [obs_val,sendcounts_obs,offsets_obs,MPI.DOUBLE])     


    ###Start conjugate gradients
    comp_CFAC = comp_Qinv**-0.5

    
        # 
    if Gt is not None:
        comp_rest_loc = Gt.dot(obs_invnoise2 * obs_val) * comp_CFAC[split_waves[rank]:split_waves[rank+1]]
        if rank==0:  comp_rest = numpy.empty((len(comp_Qinv)),dtype='f8')
        else: comp_rest = None
        sendcounts_wave = numpy.array(comm.allgather(len(comp_rest_loc)))
        offsets_wave = numpy.zeros(size)
        offsets_wave[1:]=numpy.cumsum(sendcounts_wave)[:-1]
        comm.Gatherv(sendbuf=comp_rest_loc, recvbuf=(comp_rest, sendcounts_wave), root=0)
        comp_p_loc = 1*comp_rest_loc
        comp_Ap_loc = 0*comp_rest_loc
    else:
        comp_rest = G.T.dot(obs_invnoise2_loc * obs_val_loc) * comp_CFAC
        if comm is not None:
            comm.barrier()
            comp_rest = comm.reduce(comp_rest, op=MPI.SUM, root=0)
            comm.barrier()


    if rank==0: 
        comp_p = 1*comp_rest
        comp_x = numpy.zeros_like((comp_rest))

        rest = numpy.inner(comp_rest, comp_rest)
        rest0 = +rest
    else:
        rest=None
        rest0=None
        comp_p=None
        comp_x=None
    if comm is not None:
        rest = comm.bcast(rest, root=0)
        rest0 = comm.bcast(rest0, root=0)
        comp_p = comm.bcast(comp_p, root=0)

    
    if Gt is not None:
        y = numpy.empty((len(obs_val)),dtype='f8')
        comp_Ap = numpy.empty((len(comp_Qinv)),dtype='f8')
        if rank==0: cvec = numpy.empty((len(comp_Qinv)),dtype='f8')
        else: cvec = None
    itr = int(-1)

    while ((rest / rest0 > params_algo['EPSPILLON_REST']) & (itr < params_algo['NITER'])):
        itr += 1
        if rank==0: logging.info('ITR and rest/rest0 : %s  %s %s     ', str(itr), str(rest / rest0),'             ')
        ###########################################
        # Compute A*p
        ###########################################


        if Gt is not None:  

            y_loc = G.dot(comp_p * comp_CFAC) * obs_invnoise2_loc
            comm.Allgatherv(y_loc, [y,sendcounts_obs,offsets_obs,MPI.DOUBLE])
            cvec_loc = Gt.dot(y)
            comp_Ap_loc[:] =  comp_p_loc * comp_Qinv[iw0:iw1] * comp_CFAC[iw0:iw1]**2 + cvec_loc * comp_CFAC[iw0:iw1]
        else:
            cvec = G.T.dot(G.dot(comp_p * comp_CFAC) * obs_invnoise2_loc)
            if comm is not None:
                comm.barrier()
                cvec = comm.reduce(cvec, op=MPI.SUM, root=0)
                comm.barrier()

            if rank==0:  comp_Ap =  comp_p * comp_Qinv * comp_CFAC**2 + cvec * comp_CFAC

        if itr >0: rest = +rest2

        if Gt is not None:
            tmp_loc = numpy.inner(comp_p_loc, comp_Ap_loc)
            tmp=comm.allreduce(tmp_loc, op=MPI.SUM)
            alphak = rest / tmp
        else:
            if rank==0: 
                tmp = numpy.inner(comp_p, comp_Ap)
                alphak = rest / tmp

        ###########################################
        # New state
        ###########################################
        if rank==0: comp_x += alphak * comp_p

        # ###########################################
        # New direction of descent
        ###########################################
        if Gt is not None:
            rest2_loc = numpy.inner(comp_rest_loc - alphak * comp_Ap_loc, comp_rest_loc - alphak * comp_Ap_loc)
            rest2=comm.allreduce(rest2_loc, op=MPI.SUM)
            betak = rest2 / rest
        else:
            if rank==0:
                rest2 = numpy.inner(comp_rest - alphak * comp_Ap, comp_rest - alphak * comp_Ap)
                betak = rest2 / rest

        # Loop updates
        if Gt is not None:
            comp_p_loc *= betak
            comp_p_loc += comp_rest_loc - alphak * comp_Ap_loc 
            comp_rest_loc += -alphak * comp_Ap_loc
            comm.Allgatherv(comp_p_loc, [comp_p,sendcounts_wave,offsets_wave,MPI.DOUBLE])
        else:
            if rank==0:
                comp_p *= betak
                comp_p += comp_rest - alphak * comp_Ap 
                comp_rest += -alphak * comp_Ap      
            else:
                comp_p = None
                comp_rest = None
                rest2 = None
            if comm is not None:
                rest2= comm.bcast(rest2, root=0)
                comp_p = comm.bcast(comp_p, root=0)
                comp_rest = comm.bcast(comp_rest, root=0)
          


    if rank==0:
        logging.info('last itr: %s at rest ratio: %s',itr, rest/rest0)
        return comp_x * comp_CFAC
    else:
        return None

