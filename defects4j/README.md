### Using defects4j -- v3.0.1
> 854 bugs from 17 Java repos.
> Targetting only active bugs

To pull buggy/fixed samples:
1. **Setup defects4j**
    1. Clone `git clone git@github.com:rjust/defects4j.git`
    2. Build docker image with d4j Dockerfile
    3. Bind ~/defects4j directory from this repo into the container

2. Run `python3 pull.py` to pull jsonl files per project. 
    1. Results are in defects4j/data





**Notes**: D4J goes from fixed to buggy. Patches are reversed (they introduce the bug).