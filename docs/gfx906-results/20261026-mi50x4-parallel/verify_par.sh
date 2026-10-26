#!/bin/sh
cd ~/strata-deploy
echo "== eqcheck 8"; python3 eqcheck.py 8 200
echo "== det"; python3 det.py 2>&1 | tail -1
echo "== long 820000"; python3 longctx.py 820000 300
echo "== mem"; free -g | head -2
echo DONE
