R=/scratch/ql44/we2614/mts-spi-study/representation/empirical
C=\$R/character-axes-261102
for pair in "nt:\$R/character-transfer-261031/neurotycho-primary" "nt_surr_m200:\$C/surr-neurotycho" "nt_surr_m1000:\$C/surr-neurotycho-marginal1000" "nt_surr_spec:\$C/surr-neurotycho-spectrum" "prop:\$C/propofol" "prop_surr_m:\$C/surr-propofol-marginal" "prop_surr_spec:\$C/surr-propofol-spectrum" "lemon:\$C/lemon" "lemon_surr_m:\$C/surr-lemon-marginal" "lemon_surr_spec:\$C/surr-lemon-spectrum" "adftd:\$C/ad-ftd" "adftd_surr_m:\$C/surr-ad-ftd-marginal" "adftd_surr_spec:\$C/surr-ad-ftd-spectrum"; do
  n=\${pair%%:*}; d=\${pair#*:}
  [ -s banks/\$n.npz ] && continue
  ssh -o BatchMode=yes we2614@gadi.nci.org.au "module load python3/3.12.1 >/dev/null 2>&1; OMP_NUM_THREADS=1 ~/mts-spi-study-cluster/.venv/bin/python -c 'import base64;exec(base64.b64decode(\"$B64\"))' \$d/bank.npz" > banks/\$n.npz.tmp && mv banks/\$n.npz.tmp banks/\$n.npz && echo "\$n \$(du -h banks/\$n.npz | cut -f1)"
done
echo done
