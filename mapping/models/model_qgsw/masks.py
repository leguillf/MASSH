import jax
import jax.numpy as jnp
from tools import avg_pool2d, pad_replicate

class Masks:
    """
    Masks for staggered grid used in Shallow-water discretization.
    The variables are:
     - w: vorticity
     - h: layer thickness
     - u: x-axis velocity
     - v: y-axis velocity

        y
        ^

        :           :
        w-----v-----w..
        |           |
        |           |
        u     h     u
        |           |
        |           |
        w-----v-----w..   > x

    """

    def __init__(self, mask_hgrid):
        """
        Computes automatically several masks given the binary mask on the h grid.
        as well as the irregulare boundary points
        Example of mask:
            Given the following 4_x3 domain:

                ^

                x----o---- ---- ----
                |    |XXXX|XXXX|XXXX|
                |    |XXXX|XXXX|XXXX|
                x----o----o----o----x
                |    |    |    |    |
                |    |    |    |    |
                x----o---- ---- ----x
                |XXXX|    |    |    |
                |XXXX|    |    |    |
                 ----x----x----x----x   >

            The mask_h would be:
                mask_h = torch.ones(4,3)
                mask_h[0,0] = 0
                mask_h[1:,2] = 0

            Amoung boundary points (x and o) , x are regular boundary points,
            i.e. on the rectangle boundary, and o are irregular boundary points.


        Parameters
            - mask_hgrid: float/double Tensor, shape (nx, ny), binary values
        """

        mtype = mask_hgrid.dtype
        mshape = mask_hgrid.shape

        self.h = mask_hgrid.reshape((1,)*(4-len(mshape)) + mshape)

        # Compute u, v, w, psi masks
        self.u = avg_pool2d(self.h, (2,1), stride=(1,1), padding=(1,0)) > 3/4
        self.v = avg_pool2d(self.h, (1,2), stride=(1,1), padding=(0,1)) > 3/4
        self.w = avg_pool2d(self.h, (2,2), stride=(1,1), padding=(1,1)) > 1/8
        self.psi = avg_pool2d(self.h, (2,2), stride=(1,1), padding=(1,1)) > 7/8

        # Logical negations
        self.not_h = ~self.h.astype(bool)
        self.not_u = ~self.u; self.not_v = ~self.v
        self.not_w = ~self.w; self.not_psi = ~self.psi

        # Vorticity vertical and horizontal boundary
        w_1neighbor_v = avg_pool2d(self.v.astype(mtype), (2,1), stride=(1,1),
                                     padding=(1,0), divisor_override=1) == 1
        w_1neighbor_u = avg_pool2d(self.u.astype(mtype), (1,2), stride=(1,1),
                                        padding=(0,1), divisor_override=1) == 1
        self.w_vertical_bound = self.w * (w_1neighbor_v & ~w_1neighbor_u)
        self.w_horizontal_bound = self.w * (w_1neighbor_u & ~w_1neighbor_v)
        self.w_cornerout_bound = self.w * (w_1neighbor_u & w_1neighbor_v) 
        self.w_valid = self.w * ((~self.w_vertical_bound & ~self.w_horizontal_bound) & ~self.w_cornerout_bound)

        # Irregular boundary indices
        self.psi_irrbound_xids, self.psi_irrbound_yids = jnp.where(
                self.not_psi[0,0,1:-1,1:-1] & (avg_pool2d(self.psi.astype(mtype), (3,3), stride=(1,1))[0,0] > 1/18)
                )

        # h-stencil in direction x on u-grid
        self.u_sten_hx_eq2 = \
            ((avg_pool2d(self.h.astype(mtype), (2,1), stride=(1,1), padding=(1,0)) > 3/4) & (avg_pool2d(self.h.astype(mtype), (4,1), stride=(1,1), padding=(2,0)) < 7/8)) & self.u

        self.u_sten_hx_gt4 = ~self.u_sten_hx_eq2 & self.u
        self.u_sten_hx_eq4 = (avg_pool2d(self.h.astype(mtype), (6,1), stride=(1,1), padding=(3,0)) < 11/12) & self.u_sten_hx_gt4
        self.u_sten_hx_gt6 = ~self.u_sten_hx_eq4 & self.u_sten_hx_gt4


        # w-stencil in direction y on u-grid
        self.u_sten_wy_eq2 = \
            ((avg_pool2d(self.w.astype(mtype), (1,2), stride=(1,1), padding=(0,0)) > 3/4) & (avg_pool2d(self.w.astype(mtype), (1,4), stride=(1,1), padding=(0,1)) < 7/8)) & self.u

        self.u_sten_wy_gt4 = ~self.u_sten_wy_eq2 & self.u
        self.u_sten_wy_eq4 = (avg_pool2d(self.w.astype(mtype), (1,6), stride=(1,1), padding=(0,2)) < 11/12) & self.u_sten_wy_gt4
        self.u_sten_wy_gt6 = ~self.u_sten_wy_eq4 & self.u_sten_wy_gt4


        # h-stencil in direction y on v-grid
        self.v_sten_hy_eq2 = \
                (avg_pool2d(self.h.astype(mtype), (1,2), stride=(1,1), padding=(0,1)) > 3/4) & (avg_pool2d(self.h.astype(mtype), (1,4), stride=(1,1), padding=(0,2)) < 7/8) & self.v
        self.v_sten_hy_gt4 = ~self.v_sten_hy_eq2 & self.v
        self.v_sten_hy_eq4 = (avg_pool2d(self.h.astype(mtype), (1,6), stride=(1,1), padding=(0,3)) < 11/12) & self.v_sten_hy_gt4
        self.v_sten_hy_gt6 = ~self.v_sten_hy_eq4 & self.v_sten_hy_gt4


        # w-stencil in direction x on v-grid
        self.v_sten_wx_eq2 = \
            ((avg_pool2d(self.w.astype(mtype), (2,1), stride=(1,1), padding=(0,0)) > 3/4) & (avg_pool2d(self.w.astype(mtype), (4,1), stride=(1,1), padding=(1,0)) < 7/8)) & self.v
        self.v_sten_wx_gt4 = ~self.v_sten_wx_eq2 & self.v
        self.v_sten_wx_eq4 = (avg_pool2d(self.w.astype(mtype), (6,1), stride=(1,1), padding=(2,0)) < 11/12) & self.v_sten_wx_gt4
        self.v_sten_wx_gt6 = ~self.v_sten_wx_eq4 & self.v_sten_wx_gt4


        # convert masks to correct data type
        self.h = self.h.astype(mtype)
        self.u = self.u.astype(mtype)
        self.v = self.v.astype(mtype)
        self.w = self.w.astype(mtype)
        self.w_vertical_bound = self.w_vertical_bound.astype(mtype)
        self.w_horizontal_bound = self.w_horizontal_bound.astype(mtype)
        self.w_valid = self.w_valid.astype(mtype)
        self.psi = self.psi.astype(mtype)
        self.not_h = self.not_h.astype(mtype)
        self.not_u = self.not_u.astype(mtype)
        self.not_v = self.not_v.astype(mtype)
        self.not_w = self.not_w.astype(mtype)
        self.not_psi = self.not_psi.astype(mtype)

        self.u_sten_hx_eq2 = self.u_sten_hx_eq2.astype(mtype)
        self.u_sten_hx_eq4 = self.u_sten_hx_eq4.astype(mtype)
        self.u_sten_hx_gt4 = self.u_sten_hx_gt4.astype(mtype)
        self.u_sten_hx_gt6 = self.u_sten_hx_gt6.astype(mtype)

        self.u_sten_wy_eq2 = self.u_sten_wy_eq2.astype(mtype)
        self.u_sten_wy_eq4 = self.u_sten_wy_eq4.astype(mtype)
        self.u_sten_wy_gt4 = self.u_sten_wy_gt4.astype(mtype)
        self.u_sten_wy_gt6 = self.u_sten_wy_gt6.astype(mtype)

        self.v_sten_hy_eq2 = self.v_sten_hy_eq2.astype(mtype)
        self.v_sten_hy_eq4 = self.v_sten_hy_eq4.astype(mtype)
        self.v_sten_hy_gt4 = self.v_sten_hy_gt4.astype(mtype)
        self.v_sten_hy_gt6 = self.v_sten_hy_gt6.astype(mtype)

        self.v_sten_wx_eq2 = self.v_sten_wx_eq2.astype(mtype)
        self.v_sten_wx_eq4 = self.v_sten_wx_eq4.astype(mtype)
        self.v_sten_wx_gt4 = self.v_sten_wx_gt4.astype(mtype)
        self.v_sten_wx_gt6 = self.v_sten_wx_gt6.astype(mtype)


if __name__ == "__main__":
    import numpy as np
    import matplotlib
    matplotlib.rcParams.update({'font.size': 24})
    import matplotlib.pyplot as plt

    n = 6
    mask_1 = torch.ones(n,n)

    mask_2 = torch.ones(n,n)
    mask_2[1,0] = 0.
    mask_2[n-1,2] = 0.
    mask_2[0,n-2] = 0.
    mask_2[1,n-2] = 0.
    mask_2[0,n-1] = 0.
    mask_2[1,n-1] = 0.
    mask_2[2,n-1] = 0.

    plt.ion()
    for mask, title in [(mask_1, 'rect. domain'),
                        (mask_2, 'non-rect. domain')]:
        masks = Masks(mask)

        for stencil_var in ['h', 'w']:
            f,a =plt.subplots(figsize=(14,9))
            f.suptitle(f'{title}, {stencil_var}-stencil masks')
            a.imshow(mask.T, origin='lower', cmap='Greys_r', interpolation=None, vmin=-1)
            a.set_xticks(np.arange(-0.5,n+0.5)), a.set_yticks(np.arange(-0.5,n+0.5))
            a.grid()

            # h mask
            size = 90
            size2 = 130
            q_xmin, q_ymin = 0, 0
            mask_h_ids = torch.argwhere(masks.h.squeeze())
            a.scatter(q_xmin+mask_h_ids[:,0], q_ymin+mask_h_ids[:,1],
                        s=size, marker='o', label='h', color='mediumseagreen')

            # w, psi plain and boundary mask
            w_xmin, w_ymin = -.5, -.5
            # w
            mask_w_ids = torch.argwhere(masks.w.squeeze())
            a.scatter(w_xmin+mask_w_ids[:,0], w_ymin+mask_w_ids[:,1],
                    s=size, marker='s', label='$w$', color='pink')
            mask_w_hb_ids = torch.argwhere(masks.w_horizontal_bound.squeeze())
            a.scatter(w_xmin+mask_w_hb_ids[:,0], w_ymin+mask_w_hb_ids[:,1],
                    s=2*size, marker='4', label='$w$ horiz', color='red')
            mask_w_vb_ids = torch.argwhere(masks.w_vertical_bound.squeeze())
            a.scatter(w_xmin+mask_w_vb_ids[:,0], w_ymin+mask_w_vb_ids[:,1],
                    s=2*size, marker='2', label='$w$ vert', color='green')
            mask_w_cob_ids = torch.argwhere(masks.w_cornerout_bound.squeeze())
            a.scatter(w_xmin+mask_w_cob_ids[:,0], w_ymin+mask_w_cob_ids[:,1],
                    s=2*size, marker='x', label='$w$ c_out', color='cyan')
            mask_w_valid_ids = torch.argwhere(masks.w_valid.squeeze())
            a.scatter(w_xmin+mask_w_valid_ids[:,0], w_ymin+mask_w_valid_ids[:,1],
                    s=2*size, marker='*', label='$w$ valid', color='cyan', alpha=0.4)
            # psi
            mask_psi_ids = torch.argwhere(masks.psi.squeeze())
            a.scatter(w_xmin+mask_psi_ids[:,0], w_ymin+mask_psi_ids[:,1],
                    s=size/2, marker='D', label='$\\psi$', color='brown', alpha=0.5)
            a.scatter(w_xmin+1+masks.psi_irrbound_xids,w_ymin+1+masks.psi_irrbound_yids,
                    s=size, marker='o', label='$\\mathcal{I}$', color='purple', alpha=0.5)

            if stencil_var == 'h':
                u_xmin, u_ymin = -.5, 0
                mask_u_ids = torch.argwhere(masks.u_sten_hx_eq2.squeeze())
                a.scatter(u_xmin+mask_u_ids[:,0], u_ymin+mask_u_ids[:,1],
                            s=size2, marker='>', label='u_sten_hx_eq2', color='lightblue')
                mask_u_ids = torch.argwhere(masks.u_sten_hx_eq4.squeeze())
                a.scatter(u_xmin+mask_u_ids[:,0], u_ymin+mask_u_ids[:,1],
                            s=size2, marker='>', label='u_sten_hx_eq4', color='cornflowerblue')
                mask_u_ids = torch.argwhere(masks.u_sten_hx_gt6.squeeze())
                a.scatter(u_xmin+mask_u_ids[:,0], u_ymin+mask_u_ids[:,1],
                            s=size2, marker='>', label='u_sten_hx_gt6', color='navy')

                v_xmin, v_ymin = 0, -.5
                mask_v_ids = torch.argwhere(masks.v_sten_hy_eq2.squeeze())
                a.scatter(v_xmin+mask_v_ids[:,0], v_ymin+mask_v_ids[:,1],
                            s=size2, marker='^', label='v_sten_hy_eq2', color='gold')
                mask_v_ids = torch.argwhere(masks.v_sten_hy_eq4.squeeze())
                a.scatter(v_xmin+mask_v_ids[:,0], v_ymin+mask_v_ids[:,1],
                            s=size2, marker='^', label='v_sten_hy_eq4', color='darkorange')
                mask_v_ids = torch.argwhere(masks.v_sten_hy_gt6.squeeze())
                a.scatter(v_xmin+mask_v_ids[:,0], v_ymin+mask_v_ids[:,1],
                            s=size2, marker='^', label='v_sten_hy_gt6', color='firebrick')
            elif stencil_var == 'w':
                u_xmin, u_ymin = -.5, 0
                mask_u_ids = torch.argwhere(masks.u_sten_wy_eq2.squeeze())
                a.scatter(u_xmin+mask_u_ids[:,0], u_ymin+mask_u_ids[:,1],
                            s=size2, marker='>', label='u_sten_wy_eq2', color='lightblue')
                mask_u_ids = torch.argwhere(masks.u_sten_wy_eq4.squeeze())
                a.scatter(u_xmin+mask_u_ids[:,0], u_ymin+mask_u_ids[:,1],
                            s=size2, marker='>', label='u_sten_wy_eq4', color='cornflowerblue')
                mask_u_ids = torch.argwhere(masks.u_sten_wy_gt6.squeeze())
                a.scatter(u_xmin+mask_u_ids[:,0], u_ymin+mask_u_ids[:,1],
                            s=size2, marker='>', label='u_sten_wy_gt6', color='navy')

                v_xmin, v_ymin = 0, -.5
                mask_v_ids = torch.argwhere(masks.v_sten_wx_eq2.squeeze())
                a.scatter(v_xmin+mask_v_ids[:,0], v_ymin+mask_v_ids[:,1],
                            s=size2, marker='^', label='v_sten_wx_eq2', color='gold')
                mask_v_ids = torch.argwhere(masks.v_sten_wx_eq4.squeeze())
                a.scatter(v_xmin+mask_v_ids[:,0], v_ymin+mask_v_ids[:,1],
                            s=size2, marker='^', label='v_sten_wx_eq4', color='darkorange')
                mask_v_ids = torch.argwhere(masks.v_sten_wx_gt6.squeeze())
                a.scatter(v_xmin+mask_v_ids[:,0], v_ymin+mask_v_ids[:,1],
                            s=size2, marker='^', label='v_sten_wx_gt6', color='firebrick')

            f.legend(loc='upper left')
            plt.setp(a.get_xticklabels(), visible=False)
            plt.setp(a.get_yticklabels(), visible=False)
