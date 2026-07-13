# KaHyPar configuration

`cut_rKaHyPar_sea20.ini` is vendored from the upstream KaHyPar repository so
clean-mainline runs do not depend on a machine-specific configuration path.

- Source: https://github.com/kahypar/kahypar/blob/master/config/cut_rKaHyPar_sea20.ini
- Upstream project: https://github.com/kahypar/kahypar
- Upstream license: MIT
- Runtime package on the training server: `kahypar==1.3.7`

Install it only in the Linux training environment:

```bash
python -m pip install kahypar==1.3.7
```

The main `requirements.txt` is UTF-16 and remains oriented toward the shared
Windows/Linux project dependencies, so this optional native dependency is kept
in this dedicated runbook instead of being installed on every developer host.

The adapter loads this preset first and then explicitly sets K, epsilon, and
seed. This order is required because the preset contains `seed=-1`.
