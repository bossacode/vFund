import mosek
import numpy as np


class MipOptimizer:
    def __init__(self, k, fcb, fcs, max_w, eps):
        """_summary_

        Args:
            k (int): Number of stocks to include in our portfolio.
            fcb (numpy.ndarray): Vector containing fractional cost of buying one unit of each stock. Array of size [n, ]. 
            fcs (numpy.ndarray): Vector containing fractional cost of selling one unit of each stock. Array of size [n, ].
            max_w (numpy.ndarray): Vector containing maximum values that each weight can have. Array of size [n, ].
            eps (float): Upper bound for L1 norm between previous weight and weight in current window.
        """
        self.k = k
        self.fcb = fcb
        self.fcs = fcs
        self.max_w = max_w.tolist()
        self.eps = eps


    def optimize(self, x_return, y_return, scale, w_prev):
        """
        variables: 
        1. w: weight corresponding to each stock in the universe. n variables.
        2. s: binary variable indicating whether each stock is included in the portfolio. n variables.
        3. delta: delta_i = |w_prev_i - w_i|. n variables.
        4. aux: auxiliary variable. 1 variable.

        Args:
            x_return (numpy.ndarray): Array of size [n, t].
            y_return (numpy.ndarray): Array of size [t, ].
            scale (float): Factor to scale y.
            w_prev (numpy.ndarray): Weight vector of previous window. Array of size [n, ].
        """
        n, t = x_return.shape  # [number of stocks in the universe, window size (number of time points)].
        y_return = scale * y_return   # overestimate y to reflect loss caused by transaction cost.
        inf = 0.0       # since the value of infinity is ignored, we define it solely for symbolic purposes
        init_window = (w_prev.nonzero()[0].size == 0)  # flag indicating whether it's the initial window where w_prev doesn't exist (w_prev is represented as a zero vector)
        assert 1 <= self.k and self.k <= n
        assert scale >= 1.0
        assert w_prev.shape == (n,)

        # create empty task object
        with mosek.Task() as task:  # task object represents all the data (inputs, outputs, parameters, information items etc.) associated with one optimization problem
            # variables: w, s, delta, aux
            
            if init_window:
                # don't use variable delta for initial window
                numvar = 2*n + 1
                numcon = n + 2
            else:
                numvar = 3*n + 1    # number of variables
                numcon = 3*n + 3    # number of constraints
            task.appendvars(numvar) # append "numvar" variables which are initially fixed at 0
            task.appendcons(numcon) # append "numcon" empty constraints which initially have no bounds

            # objective: minimize c^T (w, s, delta, aux), which is equivalent to minimizing aux given c = (0, ... ,0, 1)
            task.putclist(subj=range(numvar),
                        val=[0.0]*(numvar-1) + [1.0])   # set objective coefficients c
            task.putobjsense(mosek.objsense.minimize)   # input the objective sense (minimize/maximize)

            # bound keys for variables
            bk_var = [mosek.boundkey.ra]*(numvar-1) + [mosek.boundkey.fr]
            # bound values for variables: 
            # 1. 0 <= w_i <= max_w_i
            # 2. s_i in {0,1}
            # 3. 0 <= delta_i <= 1
            # 3. -inf <= aux <= inf
            lb_var = [0.0]*(numvar-1) + [-inf]                  # lower bound  ################### shouldn't the lower bound for t be restricted to 0?
            # ub_var = [1.0]*(numvar-1) + [+inf]      # upper bound
            ub_var = self.max_w + [1.0]*(numvar-n-1) + [+inf]   # upper bound
            # set bounds on variables
            task.putvarboundlist(sub=range(numvar), bkx=bk_var, blx=lb_var, bux=ub_var)
            # set integer constraints on variables s
            task.putvartypelist(subj=range(n, 2*n), vartype=[mosek.variabletype.type_int]*n)
            
            # linear constraint matrix A (sparse format)
            acol_index = [] # list to store column index of non-zero values
            aval = []       # list to store non-zero values
            # linear constraint 1: 1^T w = 1
            acol_index.append(range(n))
            aval.append([1.0]*n)
            # linear constraint: 1^T s <= k
            acol_index.append(range(n, 2*n))
            aval.append([1.0]*n)
            # linear constraint 3: w_i - s_i <= 0   ######################### shouldn't this have a lower bound?
            for i in range(n):
                acol_index.append([i, i+n])
                aval.append([1.0, -1.0])

            # turnover constraint doesn't apply for initial window
            if not init_window:
                # linear constraint 4: -w_prev_i <= -w_i + delta_i ################# shouldn't this have upper bound 2?
                for i in range(n):
                    acol_index.append([i, i+(2*n)])
                    aval.append([-1.0, 1.0])
                # linear constraint 5: w_prev_i <= w_i + delta_i   ################## shouldn't this have upper bound 2?
                for i in range(n):
                    acol_index.append([i, i+(2*n)])
                    aval.append([1.0, 1.0])
                # linear constraint 6: 0 <= sum(delta_i) <= eps
                acol_index.append(range(2*n, 3*n))
                aval.append([1.0]*n)
            
            # set constraint matrix A
            for i in range(numcon):
                task.putarow(i=i, subi=acol_index[i], vali=aval[i])
            
            if init_window:
                # bound keys for constraints
                bk_con = [mosek.boundkey.fx, mosek.boundkey.ra] + [mosek.boundkey.up]*n
                # bound values for constraints
                lb_con = [1.0, 1.0] + [-inf]*n      # lower bound
                ub_con = [1.0, self.k] + [0.0]*n    # upper bound
            else:
                bk_con = [mosek.boundkey.fx, mosek.boundkey.ra] + [mosek.boundkey.up]*n + [mosek.boundkey.lo]*(2*n) + [mosek.boundkey.ra]
                lb_con = [1.0, 1.0] + [-inf]*n + (-w_prev).tolist() + w_prev.tolist() + [0.0]
                ub_con = [1.0, self.k] + [0.0]*n + [+inf]*(2*n) + [self.eps]
            # set bounds on constraints
            task.putconboundlist(sub=range(numcon),
                                bkc=bk_con,
                                blc=lb_con,
                                buc=ub_con)

            # conic constraints
            task.appendafes(t+1)    # append empty affine expression rows for affine expression storage
            
            # F matrix
            Frow_index = [0]
            for i in range(1, t+1):
                Frow_index.extend([i]*n)
            Fcol_index = [numvar-1] + list(range(n))*t
            Fval = [1] + x_return.transpose().flatten().tolist()
            task.putafefentrylist(Frow_index, Fcol_index, Fval) # fill in F

            # g vector
            task.putafeglist(range(1,t+1), -y_return)  # fill in g

            quadcone = task.appendquadraticconedomain(t+1)  # create domain (which cone to use)
            task.appendacc(quadcone, range(t+1), None)      # append conic constraint
            
            task.optimize() # solve

            ###############################################################
            # this part needs update
            xx = task.getxx(mosek.soltype.itr)
            if task.getsolsta(mosek.soltype.itr) == mosek.solsta.optimal:
                print("Solution: {xx}".format(xx=list(xx)))
            ###############################################################


    @staticmethod
    def predict(x_return, w):
        """_summary_

        Args:
            x_return (numpy.ndarray): Array of size [n, t].
            w (numpy.ndarray): Array of size [n, ].

        Returns:
            numpy.ndarray: Array of size [t, ].
        """
        return np.matmul(x_return.transpose(), w)


    @staticmethod
    def cal_asset(x_price, asset, w):
        """_summary_

        Args:
            x_price (numpy.ndarray): Array of size [n, t].
            asset (float): Value of total asset at start of window.
            w (numpy.ndarray): Weight vector of current window. Array of size [n, ].
        """
        stock_units = (asset * w) / x_price[:, 0]    # number of units of each stock in the portfolio
        return stock_units * x_price[:, -1]


    def cal_transcost(self, asset, w, w_prev):
        """_summary_

        Args:
            asset (float): Value of total asset at time of reblancing.
            w (numpy.ndarray): Weight vector of current window. Array of size [n, ].
            w_prev (numpy.ndarray): Weight vector of previous window. Array of size [n, ].

        Returns:
            _type_: _description_
        """
        buy = np.heaviside(w - w_prev, 0)   # indicator whether each stock was bought during rebalancing
        transcost = asset * np.matmul(self.fcb*buy - self.fcs*(1-buy), w-w_prev)
        return transcost