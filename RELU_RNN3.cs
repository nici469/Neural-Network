using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Counter_Console
{
    public class RELU_RNN3:RELU_RNN
    {
        public RELU_RNN3(int[] hiddenNodes, int T) : base(hiddenNodes, T) { }
        public bool isManyToOne = true;
        void ComputeLastLayer(int l, int t)
        {
            LoadVariables(l, t);
            dY = GetBackPropInput(l, t);
            dYn = dY * M.TanhPrime(Yn);
            dX = Wi.T() * dYn;

            dWi += dYn ^ X;
            if (t > 0)
            {
                Vector yt_1 = new Vector(YJ[l, t - 1]);// y at t-1
                dUi += dYn ^ yt_1;
            }
        }
        public override void PropagateBackward()
        {
            for (int l = L - 1; l >= 0; l--)
            {
                ObtainNodes(l);
                InitWeightGradients();
                InitLayerMarices(l);

                for (int t = incT - 1; t >= 0; t--)
                {

                    InitVariableGradient();
                    //if its the last layer and not the last time step for many-to-one case, propagate zero gradient
                    if (l == L - 1 && t < incT - 1 && isManyToOne)
                    {
                        dX = new Vector(N);
                        SaveVariableGradient(l, t);
                        continue;
                    }
                    else if (l == L - 1)
                    {//the last layer
                        ComputeLastLayer(l, t);
                        SaveVariableGradient(l, t);


                    }
                    else
                    {
                        LoadVariables(l, t);
                        dY = GetBackPropInput(l, t);
                        dYn = dY * M.ReLU_Prime(Yn);
                        dX = Wi.T() * dYn;

                        dWi += dYn ^ X;
                        if (t > 0)
                        {
                            Vector yt_1 = new Vector(YJ[l, t - 1]);// y at t-1
                            dUi += dYn ^ yt_1;
                        }


                        SaveVariableGradient(l, t);
                    }
                    
                }
                //compute new weights
                Wi += -1 * lnRate * dWi;
                Ui += -1 * lnRate * dUi;
                Bi += -1 * lnRate * dBi;
                SaveWeights(l);
            }
        }
        /// <summary>
        /// performs the forward propagation through the counter network for all the timesteps
        /// </summary>
        public override void PropagateForward()
        {
            for (int l = 0; l < L; l++)
            {
                ObtainNodes(l);
                InitLayerMarices(l);
                for (int t = 0; t < T; t++)
                {
                    InitVariableVectors(l, t);
                    X.data = GetInput(l, t);
                    if (l == L - 1)
                    {
                        ComputeVariables2(l, t);
                       
                    }
                    else
                    {
                        ComputeVariables(l, t);
                    }
                    
                    
                    SaveAllVariables(l, t);
                }
            }
        }
        /// <summary>
        /// compute a tanh output function for the last layer
        /// </summary>
        /// <param name="l"></param>
        /// <param name="t"></param>
        public void ComputeVariables2(int l, int t)
        {
            Vector Yt_1;
            if (t > 0)
            {
                Yt_1.data = YJ[l, t - 1];
            }
            else
            {
                Yt_1 = new Vector(D);
            }
            Yn = Wi * X + Ui * Yt_1 + Bi;
            Y = M.Tanh(Yn);
        }



    }
}
