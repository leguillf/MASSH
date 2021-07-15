import logging
import numpy
import sparse_inversion
import sys, os
from mpi4py import MPI
import pdb


def init_mpi():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm, rank, size


def main(param_file):

    config=load_parameters(param_file)
    config['RUN_NAME'] = param_file.rsplit('/', 1)[-1].rsplit('.', 1)[0]
    if config['ALGO']['USE_MPI']: 
        comm, rank, size = init_mpi()
    else: comm, rank, size = None,0,1
    run_miost(config, comm=comm, rank=rank, size=size)



def run_miost(config, comm=None, rank=0, size=1):
    FORMAT_LOG = "%(levelname)-10s %(asctime)s %(module)s." \
                     "%(funcName)s : %(message)s"
    logging.basicConfig(format=FORMAT_LOG, level=logging.INFO,
                            datefmt="%H:%M:%S")
    use_mpi = config['ALGO']['USE_MPI']

    ##################################################
    # Initialize and broadcast the grid and obs objects
    ##################################################

    if rank==0:

        logging.info('')
        logging.info('RUN_NAME:  %s', config['RUN_NAME'])

        grid = config['GRID']
        logging.info('')
        logging.info('TIME: %s to   %s  TIME Julian: %s  to %s', grid.DATE_MIN,
                    grid.DATE_MAX, grid.TIME_MIN, grid.TIME_MAX)
        obs = config['OBS']

    else:
        grid=None
        obs=None
    if use_mpi: 
        grid = comm.bcast(grid, root=0)
        obs = comm.bcast(obs, root=0)

    ##################################################
    # Load obs (on each proc)
    ##################################################

    if rank==0:
        logging.info('')
        logging.info('Loading observations...')
        logging.info('')


    obs_data = [None]*len(obs)
    obsmask = numpy.ones((len(obs)), dtype=bool)
    nobs_loc = 0
    for ko in range(len(obs)):
        obs_data[ko] = obs[ko].get_obs([grid.LON_MIN, grid.LON_MAX, grid.LAT_MIN, grid.LAT_MAX, grid.TIME_MIN, grid.TIME_MAX],
                 chunk=rank, nchunks=size)
        if obs_data[ko][3]['nobs'] == 0: obsmask[ko] = False
        else: logging.debug('%s %s obs loaded from proc %s', obs_data[ko][3]['nobs'], obs[ko].name, rank)
        if use_mpi: comm.barrier()
        nobs_loc += obs_data[ko][3]['nobs']

    for ko in range(len(obs)):
        if use_mpi: obs[ko].nobs_tot = comm.reduce(obs_data[ko][3]['nobs'], op=MPI.SUM, root=0)
        else: obs[ko].nobs_tot = obs_data[ko][3]['nobs']
        if rank == 0: logging.info('%s %s obs loaded', obs[ko].nobs_tot, obs[ko].name)

    if use_mpi: obsmask = comm.allreduce(obsmask, op=MPI.SUM)
    
    obs = list(numpy.asarray(obs)[obsmask == True])
    obs_data = list(numpy.asarray(obs_data)[obsmask == True])
    if rank ==0:  
        logging.info('')
        logging.info('Loading observations Done')
        logging.info('')
    if use_mpi: comm.barrier()
    

    ##################################################
    # Initialize physical and obs-related components
    ##################################################

    if rank==0:
        comp=[]
        # Physical components
        kc=-1
        comp += config['PHYS_COMP']
        for kc in range(len(comp)):
            comp[kc].set_domain(grid)
            comp[kc].set_basis()
            comp[kc].flag_obs_datasets(obs)
            #comp[kc].id = kc

        # Observation components (BGLO, roll for SWOT, ...)  
        comp += config['OBS_COMP']
        for koc in numpy.arange(0,len(comp))[kc+1:]:
            comp[koc].obs_datasets = []
            for ko in range(len(obs)):
                if obs[ko].name in  comp[koc].obs_dataset_names: 
                    comp[koc].obs_datasets += [ko]
        is_koc = numpy.ones((len(comp)), dtype=bool)
        is_koc[:len(config['PHYS_COMP'])]=False
        compmask = numpy.ones((len(comp)), dtype=bool)
        for kc in range(len(comp)):
            if len(comp[kc].obs_datasets)==0: compmask[kc]=False
        comp = list(numpy.asarray(comp)[compmask == True])
        is_koc = list(numpy.asarray(is_koc)[compmask == True])

    else: 
        comp=None
        is_koc=None
    if use_mpi: 
        comp = comm.bcast(comp, root=0)
        is_koc = comm.bcast(is_koc, root=0)

    for kc in numpy.arange(len(comp))[is_koc]:
        comp[kc].set_domain([obs[ko] for ko in comp[kc].obs_datasets] ,[obs_data[ko] for ko in comp[kc].obs_datasets], grid, comm)
        comp[kc].set_basis()

    if use_mpi: comm.barrier()

    ##################################################
    # Print status of obs and comps involved
    ##################################################

    if rank==0:

        logging.info('')
        logging.info('OBSERVATION DATASETS :')
        for ko in range(len(obs)):
            logging.info(' %s :   %s  nature: %s  size: %s', ko, obs[ko].name, obs[ko].nature, obs[ko].nobs_tot)
        logging.info('')
        logging.info('COMPONENTS :')
        nwave = 0
        for kc in range(len(comp)):
            logging.info(' %s :   %s  write:%s obs datasets affected: %s  size: %s', kc, comp[kc].name, comp[kc].write, comp[kc].obs_datasets, comp[kc].nwave)
            nwave += comp[kc].nwave
        logging.info('')

    ##################################################
    # Fill Qinv
    ##################################################
        logging.info('Fill Qinv ...')
        comp_Qinv = numpy.zeros((nwave))
        iwave=0
        for kc in range(len(comp)):
            comp_Qinv[iwave:iwave+comp[kc].nwave] = comp[kc].set_basis(return_qinv=True)
            iwave += comp[kc].nwave   
    else:
        comp_Qinv = None
        nwave=None
    if use_mpi: comp_Qinv = comm.bcast(comp_Qinv, root=0)
    if use_mpi: nwave = comm.bcast(nwave, root=0)

    ##################################################
    # Fill y (obs_val vector) and Rinv (obs_invnoise2)
    ##################################################
    if rank==0: logging.info('Fill y (obs_val vector) and Rinv (obs_invnoise2) ...')
    obs_val_loc = numpy.zeros((nobs_loc))
    obs_invnoise2_loc = numpy.zeros((nobs_loc))
    iobs=0
    for ko in range(len(obs)):
        if obs_data[ko][3]['nobs']>0:
            obs_val_loc[iobs:iobs+obs_data[ko][3]['nobs']] = obs_data[ko][0]
            obs_invnoise2_loc[iobs:iobs+obs_data[ko][3]['nobs']] = obs_data[ko][1]**-2
            iobs += obs_data[ko][3]['nobs']
    # To do: then remove obs_data


    ##################################################
    # Fill G
    ##################################################
    Gdata= sparse_inversion.build_gmatrix(comp, obs, obs_data,  config['ALGO'], comm=comm, rank=rank, size=size)

    ##################################################
    # PCG inversion
    ##################################################
    eta = sparse_inversion.solve_pcg(Gdata, comp_Qinv, obs_invnoise2_loc, obs_val_loc,  config['ALGO'], rank=rank, comm=comm)

    if comm is not None: eta = comm.bcast(eta, root=0)
    if comm is not None: config = comm.bcast(config, root=0)
    obs_val_loc_expl = Gdata[0].dot(eta[:])
    file_pickle = config['PATH']['OUTPUT']+"/obs_val_loc_expl_proc"+str(rank)+".p"
    import pickle
    pickle.dump([obs_val_loc_expl], open(file_pickle , "wb" ) )
    logging.info('Miost step 0 obs_val_loc_expl size %s written %s',len(obs_val_loc_expl), file_pickle)

    if rank==0:
        iwave=0
        data_comp = [None]*len(comp)
        for kc in range(len(comp)):
            data_comp[kc] = eta[iwave:iwave+comp[kc].nwave]
            iwave += comp[kc].nwave
    else: data_comp = None
    if comm is not None: data_comp = comm.bcast(data_comp, root=0)
    if comm is not None: config = comm.bcast(config, root=0)

    if comm is not None: comm.barrier()
    if rank==0: logging.info('Project and write the solution on the grid...')
    grid.write_outputs(comp, data_comp, config, rank=rank, size=size)




    return None




def load_parameters(file_path):
    """Load a file and parse it as a Python module."""
    if not os.path.exists(file_path):
        raise IOError('File not found: {}'.format(file_path))

    full_path = os.path.abspath(file_path)
    python_filename = os.path.basename(full_path)
    module_name, _ = os.path.splitext(python_filename)
    module_dir = os.path.dirname(full_path)
    if module_dir not in sys.path:
        sys.path.append(module_dir)

    module = __import__(module_name, globals(), locals(), [], 0)
    return module.config


if __name__ == "__main__":
    main( sys.argv[1])
