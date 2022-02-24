using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

namespace Counter_Console
{
    public class MLFileReader
    {
        
        /// <summary>
        /// The number of input variables
        /// </summary>
        private int Ni;

        /// <summary>
        /// The number of output variables
        /// </summary>
        private int No;

        /// <summary>
        /// the full path of the file containing the data to be read for ML training
        /// </summary>
        private string filePath;

        /// <summary>
        /// an array to store the lines of string read from filePath
        /// </summary>
        string[] lines;

        /// <summary>
        /// used to store the position of the start signature of the first timestep in the filePath
        /// </summary>
        int startSignPos;

        /// <summary>
        /// used to store the position of the ending signature of the first timestep in the filePath
        /// </summary>
        int firstEndSignPos;

        /// <summary>
        /// a signature string that specifies the start of a new timestep
        /// </summary>
        string startSign;

        /// <summary>
        /// a signature string that specifies the end of a timestep
        /// </summary>
        string endSign;

        /// <summary>
        /// Jagged array containig the ML input data in format [Timestep] [dataId] or
        /// [T][Nc]
        /// </summary>
        private double[][] MLInput;

        /// <summary>
        /// Jagged array containig the ML input data in format [Timestep] [dataId] or
        /// [T][Ni]
        /// </summary>
        private double[][] TOutput;
        
        
        public void Init()
        {
            

        }
    }
}
