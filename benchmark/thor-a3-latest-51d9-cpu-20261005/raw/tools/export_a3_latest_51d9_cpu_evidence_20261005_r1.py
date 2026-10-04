# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json,hashlib,tarfile,subprocess
from pathlib import Path
R=Path('/home/jwipc/experiments/cudnn-sm110-a1-a3-20260928');source=R/'a3sfdlatest51d9_r1';archive=R/'results/a3-latest-51d9-cpu-evidence-20261005-r1.tar.gz';bundle=R/'results/a3-latest-51d9-source-20261005-r1.bundle';assert not archive.exists() and not bundle.exists()
prov=json.loads((R/'results/a3-latest-51d9-native-provenance-20261005-r1.json').read_text());assert prov['native_build']=='FRESH_RC0' and prov['head']=='2fc0bc608c19c75311a405b12cc57374741f9df1';assert hashlib.sha256(Path(prov['binding']['path']).read_bytes()).hexdigest()==prov['binding']['sha256'];assert all(hashlib.sha256((source/item['path']).read_bytes()).hexdigest()==item['sha256'] for item in prov['native_input_digests'])
files=[R/'results/a3-latest-51d9-native-provenance-20261005-r1.json',R/'results/a3-latest-51d9-prebuild-20261005-r1.json',R/'tools/prepare_build_a3_latest_51d9_20261005_r1.py']
for name in ['a3-latest-51d9-native-preparation-20261005-r1','build-a3-latest-51d9-20261005-r1']:
 d=R/'results'/name;r=json.loads((d/'run.json').read_text());assert r['returncode']==0 and r['signal'] is None and r['signals_sent']==0
 files += [d/f for f in ['run.json','run.live.json','stdout.log','stderr.log']]
bg=R/'results/a3-latest-51d9-preparation-background-20261005-r1';files += [bg/f for f in ['launch.json','policy_manifest.json','driver.log']]
for path in (bg/'redirects').glob('*.json'): files.append(path)
policy=R/'tools/a3_async_policy_20261004';files += [policy/f for f in ['manifest.json','run_logged_observe.py','a3_no_signal_redirect.py','sitecustomize.py']]
manifest={'files':[{'path':str(f.relative_to(R)),'size':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files],'excluded':['model','cache','engine','.pt','binary'],'source_head':prov['head'],'upstream':prov['base'],'binding_hash_only':prov['binding']['sha256']};mf=R/'results/a3-latest-51d9-cpu-evidence-manifest-20261005-r1.json';assert not mf.exists();mf.write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(archive,'w:gz') as t:
 for f in files: t.add(f,arcname=str(f.relative_to(R)))
 t.add(mf,arcname='MANIFEST.json')
subprocess.run(['git','-C',str(source),'bundle','create',str(bundle),'c7841117c988fc09cad1df415f10e3ce7d6fa50e..codex/a3-sfd-latest-51d9-20261005-r1'],check=True)
print(json.dumps({'archive':str(archive),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'archive_bytes':archive.stat().st_size,'bundle':str(bundle),'bundle_sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),'bundle_bytes':bundle.stat().st_size,'manifest':manifest},indent=2))
