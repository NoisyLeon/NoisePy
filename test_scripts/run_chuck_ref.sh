#!/bin/bash
IDexe=/home/leon/code/Rftn.Codes.Export/IterDecon/iterdecon
CRdata=R.sac
CZdata=Z.sac
$IDexe <<END
$CRdata
$CZdata
200      * nbumps
5.0      * phase delay for result
0.001    * min error improvement to accept
2.5      * Gaussian width factor
1        * 1 allows negative bumps
0        * 0 form minimal output (1) will output lots of files
END

