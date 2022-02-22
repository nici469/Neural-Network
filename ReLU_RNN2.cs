using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Counter_Console
{
    public class RELU_RNN2:RELU_RNN
    {
        public bool isManyToOne=true;
        public RELU_RNN2(int[] hiddenNodes, int T) : base(hiddenNodes, T) { }
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
                    if (l == L - 1 && t < incT - 1&&isManyToOne)
                    {
                        dX = new Vector(N);
                        SaveVariableGradient(l, t);
                        continue;
                    }
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
                //compute new weights
                Wi += -1 * lnRate * dWi;
                Ui += -1 * lnRate * dUi;
                Bi += -1 * lnRate * dBi;
                SaveWeights(l);
            }
        }


    }
}
