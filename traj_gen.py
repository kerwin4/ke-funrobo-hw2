
import numpy as np

from funrobo_kinematics.core.trajectory_generator import MultiAxisTrajectoryGenerator, MultiSegmentTrajectoryGenerator


class CubicPolynomial():
    """
    Cubic interpolation with position and velocity boundary constraints.
    """

    def __init__(self, ndof=None):
        """
        Initialize the trajectory generator.
        """
        self.ndof = ndof

    
    def solve(self, q0, qf, qd0, qdf, T):
        """
        Compute cubic polynomial coefficients for each DOF.

        Parameters
        ----------
        q0 : array-like, shape (ndof,)
            Initial positions.
        qf : array-like, shape (ndof,)
            Final positions.
        qd0 : array-like or None, shape (ndof,)
            Initial velocities. If None, assumed zero.
        qdf : array-like or None, shape (ndof,)
            Final velocities. If None, assumed zero.
        T : float
            Total trajectory duration.
        """
        t0, tf = 0, T
        q0 = np.asarray(q0, dtype=float)
        qf = np.asarray(qf, dtype=float)
        qd0 = np.zeros_like(q0) if qd0 is None else np.asarray(qd0, dtype=float)
        qdf = np.zeros_like(q0) if qdf is None else np.asarray(qdf, dtype=float)
        
        A = np.array(
                [[1, t0, t0**2, t0**3],
                 [0, 1, 2*t0, 3*t0**2],
                 [1, tf, tf**2, tf**3],
                 [0, 1, 2*tf, 3*tf**2]
                ])

        b = np.vstack([
            q0,
            qd0,
            qf,
            qdf
        ])
        self.coeff = np.linalg.solve(A, b)
        

    def generate(self, t0=0, tf=0, nsteps=100):
        """
        Generate position, velocity, and acceleration trajectories.

        Parameters
        ----------
        t0 : float
            Start time.
        tf : float
            End time.
        nsteps : int
            Number of time samples.
        """
        t = np.linspace(t0, tf, nsteps)
        X = np.zeros((self.ndof, 3, len(t)))
        for i in range(self.ndof): # iterate through all DOFs
            c = self.coeff[:, i]

            q = c[0] + c[1] * t + c[2] * t**2 + c[3] * t**3
            qd = c[1] + 2 * c[2] * t + 3 * c[3] * t**2
            qdd = 2 * c[2] + 6 * c[3] * t

            X[i, 0, :] = q      # position
            X[i, 1, :] = qd     # velocity
            X[i, 2, :] = qdd    # acceleration

        return t, X


class QuinticPolynomial():
    """
    Quintic interpolation with position and velocity boundary constraints.
    """

    def __init__(self, ndof=None):
        """
        Initialize the trajectory generator.
        """
        self.ndof = ndof

    
    def solve(self, q0, qf, qd0, qdf, T):
        """
        Compute cubic polynomial coefficients for each DOF.

        Parameters
        ----------
        q0 : array-like, shape (ndof,)
            Initial positions.
        qf : array-like, shape (ndof,)
            Final positions.
        qd0 : array-like or None, shape (ndof,)
            Initial velocities. If None, assumed zero.
        qdf : array-like or None, shape (ndof,)
            Final velocities. If None, assumed zero.
        T : float
            Total trajectory duration.
        """
        t0, tf = 0, T
        q0 = np.asarray(q0, dtype=float)
        qf = np.asarray(qf, dtype=float)
        qd0 = np.zeros_like(q0) if qd0 is None else np.asarray(qd0, dtype=float)
        qdf = np.zeros_like(q0) if qdf is None else np.asarray(qdf, dtype=float)
        qa0 = np.zeros_like(q0)
        qaf = np.zeros_like(q0)
        
        A = np.array(
                [[1, t0, t0**2, t0**3, t0**4, t0**5],
                 [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],
                 [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3],
                 [1, tf, tf**2, tf**3, tf**4, tf**5],
                 [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
                 [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3]
                ])

        b = np.vstack([
            q0,
            qd0,
            qa0,
            qf,
            qdf,
            qaf
        ])
        self.coeff = np.linalg.solve(A, b)
        

    def generate(self, t0=0, tf=0, nsteps=100):
        """
        Generate position, velocity, and acceleration trajectories.

        Parameters
        ----------
        t0 : float
            Start time.
        tf : float
            End time.
        nsteps : int
            Number of time samples.
        """
        t = np.linspace(t0, tf, nsteps)
        X = np.zeros((self.ndof, 3, len(t)))
        for i in range(self.ndof): # iterate through all DOFs
            c = self.coeff[:, i]

            q = c[0] + c[1] * t + c[2] * t**2 + c[3] * t**3 + c[4] * t**4 + c[5] * t**5
            qd = c[1] + 2 * c[2] * t + 3 * c[3] * t**2 + 4 * c[4] * t**3 + 5 * c[5] * t**4
            qdd = 2 * c[2] + 6 * c[3] * t + 12 * c[4] * t**2 + 20 * c[5] * t**3

            X[i, 0, :] = q      # position
            X[i, 1, :] = qd     # velocity
            X[i, 2, :] = qdd    # acceleration

        return t, X

class Trapezoidal():
    """
    Trapezoidal interpolation with position and velocity boundary constraints.
    """

    def __init__(self, ndof=None):
        """
        Initialize the trajectory generator.
        """
        self.ndof = ndof

    
    def solve(self, q0, qf, qd0, qdf, T):
        """
        Compute cubic polynomial coefficients for each DOF.

        Parameters
        ----------
        q0 : array-like, shape (ndof,)
            Initial positions.
        qf : array-like, shape (ndof,)
            Final positions.
        qd0 : array-like or None, shape (ndof,)
            Initial velocities. If None, assumed zero.
        qdf : array-like or None, shape (ndof,)
            Final velocities. If None, assumed zero.
        T : float
            Total trajectory duration.
        """
        t0, tf = 0, T
        q0, qf = np.array(q0), np.array(qf)
        V_peak_min = (qf - q0) / tf
        V_peak_max = 2 * (qf - q0) / tf
        V_peak = (V_peak_max + V_peak_min) / 2
        tb = (q0 - qf + V_peak * tf) / V_peak
        a_peak = V_peak / tb
        self.vars = [t0, tf, V_peak, tb, a_peak]
        self.q0 = q0
        self.qf = qf
        
    def generate(self, t0=0, tf=0, nsteps=100):
        """
        Generate position, velocity, and acceleration trajectories.

        Parameters
        ----------
        t0 : float
            Start time.
        tf : float
            End time.
        nsteps : int
            Number of time samples.
        """
        t0, tf, V_peak, tb, a_peak = self.vars
        q0, qf = self.q0, self.qf
        t = np.linspace(t0, tf, nsteps)
        X = np.zeros((self.ndof, 3, len(t)))

        for i in range(self.ndof):
            #phases
            acc = t < tb[i]
            const = (t >= tb[i]) & (t <= tf - tb[i])
            dec = t > tf - tb[i]

            #acceleration
            X[i, 0, acc] = q0[i] + 0.5 * a_peak[i] * t[acc]**2
            X[i, 1, acc] = a_peak[i] * t[acc]
            X[i, 2, acc] = a_peak[i]

            #constant velocity
            X[i, 0, const] = (q0[i] + qf[i] - V_peak[i]*tf)/2 + V_peak[i]*t[const]
            X[i, 1, const] = V_peak[i]
            X[i, 2, const] = 0

            #deceleration
            X[i, 0, dec] = qf[i] - 0.5*a_peak[i]*tf**2 + a_peak[i]*tf*t[dec] - 0.5*a_peak[i]*t[dec]**2
            X[i, 1, dec] = a_peak[i]*(tf - t[dec])
            X[i, 2, dec] = -a_peak[i]

        return t, X
    
def main():
    ndof = 2
    method = Trapezoidal(ndof=ndof) # change for other trajectory methods
    mode = "joint"

    # --------------------------------------------------------
    # Point-to-point multi-axis trajectory generator
    # --------------------------------------------------------

    # traj = MultiAxisTrajectoryGenerator(method=method,
    #                                     mode=mode,
    #                                     ndof=ndof)
    
    # traj.solve(q0=-30, qf=60, T=1)
    # traj.generate(nsteps=20)

    # --------------------------------------------------------
    # Via point multi-axis trajectory generator
    # --------------------------------------------------------

    traj = MultiSegmentTrajectoryGenerator(method=method,
                                           mode=mode,
                                           ndof=ndof,
                                            )
    via_points = [[-30, 30], [0, 45], [30, 15], [50, -30]]

    traj.solve(via_points, T=2)
    traj.generate(nsteps_per_segment=20)
    
    
    # plotter
    traj.plot()

if __name__ == "__main__":
    main()