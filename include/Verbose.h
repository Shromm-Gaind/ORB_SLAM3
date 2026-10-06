/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós,
 * University of Zaragoza. Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of
 * Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ORB-SLAM3.
 * If not, see <http://www.gnu.org/licenses/>.
 */

 #ifndef VERBOSE_H
 #define VERBOSE_H
 
 #include <atomic>
 #include <iostream>
 #include <string>
 
 namespace ORB_SLAM3 {
 
     // Console verbosity and the run log.
     //
     // Verbose::on is the verbose flag. Informational output is written as
     //
     //     if (Verbose::on) cout << ... << endl;
     //
     // so it is skipped entirely when the flag is off. Errors and warnings are not
     // wrapped and always print. The flag is off by default; System's constructor
     // sets it from "System.Verbose: 1" in the settings file, or call
     // Verbose::SetVerbose(true) before constructing System.
     //
     // Run log: once StartLog() has been called (System's constructor does it),
     // everything written to cout / cerr still reaches the terminal and is also
     // kept in memory. SaveLog() writes that copy to a file. System::Shutdown()
     // calls it, so run.log appears on stop, next to the trajectory files.
     class Verbose {
     public:
         enum eLevel {
             VERBOSITY_QUIET = 0,
             VERBOSITY_NORMAL = 1,
             VERBOSITY_VERBOSE = 2,
             VERBOSITY_VERY_VERBOSE = 3,
             VERBOSITY_DEBUG = 4
         };
 
         static std::atomic<bool> on;
         static eLevel th;
     public:
         static void PrintMess(std::string str, eLevel lev)
         {
             if (on && lev <= th)
             {
                 std::cout << str << std::endl;
             }
         }
 
         static void SetTh(eLevel _th) { th = _th; }
 
         static void SetVerbose(bool bVerbose) { on = bVerbose; }
 
         // Start mirroring cout / cerr into the in-memory run log. Safe to call more than once.
         static void StartLog();
 
         // Write everything logged so far to filename, replacing its previous contents.
         // Does nothing if StartLog() was never called.
         static void SaveLog(const std::string& filename = "run.log");
     };
 
 } // namespace ORB_SLAM3
 
 #endif // VERBOSE_H
 