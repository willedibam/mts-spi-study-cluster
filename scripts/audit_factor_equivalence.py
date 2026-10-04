"""Certificate that the two factor classes differ only by signed channel relabeling."""
import json
import numpy as np
from scripts.scout_factor_channels import signs,covariance,CLASSES,OUT


def run():
    records=[]
    for m in [8,16,32]:
        a,b=[signs(label,m) for label in CLASSES];ca,cb=[covariance(label,m) for label in CLASSES]
        canonical_a=(a*a[0]).T;canonical_b=(b*b[0]).T;available=list(range(m));permutation=[]
        for row in canonical_b:
            k=next(i for i in available if np.array_equal(canonical_a[i],row));permutation.append(k);available.remove(k)
        permutation=np.array(permutation);polarity=b[0]/a[0,permutation]
        np.testing.assert_array_equal(b,a[:,permutation]*polarity)
        mapped=ca[np.ix_(permutation,permutation)]*polarity[:,None]*polarity[None,:]
        error=float(abs(mapped-cb).max());assert error<1e-12
        records.append(dict(M=m,permutation=permutation.tolist(),channel_signs=polarity.tolist(),max_covariance_error=error))
    report=dict(result='Exactly equivalent under channel permutation and channel-wise polarity reversal',
        identity='V_B = V_A P D, hence C_B = D P^T C_A P D for any common diagonal factor weights and strength',records=records,
        interpretation='A distinction can reflect coordinate polarity. Physical interpretation of signs requires a fixed measurement convention. This is not evidence of different intrinsic Gaussian dynamics or signed topology invariant under polarity changes.')
    (OUT/'polarity-equivalence.json').write_text(json.dumps(report,indent=2)+'\n');print(report['result'])

if __name__=='__main__':run()
