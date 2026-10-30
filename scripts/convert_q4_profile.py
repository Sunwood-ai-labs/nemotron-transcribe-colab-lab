import sys
from pathlib import Path

repo = Path('/content/nemo-speech')
sys.path.insert(0, str(repo))
from conversion import salm

# Extend only this converter process; use the standard CPU quantizer with Q4_0 tensor rules.
salm.PROFILES = (*salm.PROFILES, 'q4_0')
salm.DENSE_TYPE['q4_0'] = 'q8_0'
standard_quantizer = salm.ensure_quantizer

def ensure_standard_quantizer(*args, **kwargs):
    if kwargs.get('profile') == 'q4_0':
        kwargs['profile'] = 'iq4_nl'
    return standard_quantizer(*args, **kwargs)

salm.ensure_quantizer = ensure_standard_quantizer
salm.convert(Path(sys.argv[1]), Path(sys.argv[2]), profile='q4_0', dry_run='--dry-run' in sys.argv[3:], encoder='f16')
