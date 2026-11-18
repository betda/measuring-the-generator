# Code versions

Every version of the `clm_losses` package used in the study, zipped as `clm_losses-<version>-<code hash>.zip`. The code hash is the one each run stores in its `config.json` (`clm_losses.utils.code_hash`). The root `README.md` maps each version to the phases that ran with it. Version 0.3.8, used for the post hoc analyses, is the package at the root of the repository.

To check a version:

```bash
unzip clm_losses-0.3.4-94211b491b3f5d44.zip -d v034 && cd v034
python -c "from clm_losses.utils import code_hash; print(code_hash())"   # 94211b491b3f5d44
```
