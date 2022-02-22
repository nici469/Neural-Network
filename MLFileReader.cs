using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Counter_Console
{
    public class MLFileReader
    {
        /// <summary>
        /// The number of input variables
        /// </summary>
        public int Ni;
        /// <summary>
        /// The number of output variables
        /// </summary>
        public int No;
        /// <summary>
        /// the number of timesteps to be stored in MLInput and TOutput, or the number of timesteps
        /// to be processed
        /// </summary>
        public int T;
        /// <summary>
        /// Jagged array containig the ML input data in format [Timestep] [dataId] or
        /// [T][Nc]
        /// </summary>
        public double[][] MLInput;
        /// <summary>
        /// Jagged array containig the ML input data in format [Timestep] [dataId] or
        /// [T][Ni]
        /// </summary>
        public double[][] TOutput;
    }
}
