import json
import os
import subprocess
from pathlib import Path


def run_command(cmd):
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True
    )
    if result.returncode != 0:
        print(f"Error running command: {cmd}\nError: {result.stderr}")
    return result.stdout.strip()


def get_bug_info(project, bug_id):
    working_dir_buggy = f"/tmp/{project}_{bug_id}_buggy"
    working_dir_fixed = f"/tmp/{project}_{bug_id}_fixed"

    run_command(f"defects4j checkout -p {project} -v {bug_id}b -w {working_dir_buggy}")
    run_command(f"defects4j checkout -p {project} -v {bug_id}f -w {working_dir_fixed}")

    modified_classes = run_command(
        f"defects4j export -p classes.modified -w {working_dir_fixed}"
    ).splitlines()

    src_dir_buggy = run_command(
        f"defects4j export -p dir.src.classes -w {working_dir_buggy}"
    )
    src_dir_fixed = run_command(
        f"defects4j export -p dir.src.classes -w {working_dir_fixed}"
    )

    bug_data = {
        "project": project,
        "bug_id": bug_id,
        "classes_modified": [],
    }

    for cls in modified_classes:
        class_path_buggy = os.path.join(
            working_dir_buggy, src_dir_buggy, cls.replace(".", "/") + ".java"
        )
        class_path_fixed = os.path.join(
            working_dir_fixed, src_dir_fixed, cls.replace(".", "/") + ".java"
        )

        # Ensure the file exists before attempting to read
        if Path(class_path_buggy).is_file() and Path(class_path_fixed).is_file():
            with open(class_path_buggy, "r") as f_buggy:
                buggy_code = f_buggy.read()
            with open(class_path_fixed, "r") as f_fixed:
                fixed_code = f_fixed.read()

            # Add the class data to the bug info
            bug_data["classes_modified"].append(
                {
                    "class_name": cls,
                    "buggy_version": buggy_code,
                    "fixed_version": fixed_code,
                }
            )
        else:
            print(
                f"File {class_path_buggy} or {class_path_fixed} not found, skipping this class."
            )

    return bug_data


projects = {
    # "Chart": range(1, 27),
    # "Cli": [i for i in range(1, 41) if i != 6],
    # "Closure": [i for i in range(1, 63)] # dropping non active bugs 
    # + [i for i in range(64, 93)]
    # + [i for i in range(94, 177)],
    # "Codec": range(1, 19),
    # "Collections": range(1, 29),
    # "Compress": range(1, 48),
    # "Csv": range(1, 17),
    # "Gson": range(1, 19),
    # "JacksonCore": range(1, 27),
    "JacksonDatabind": [i for i in range(1, 65)] # dropping non active bugs 
    + [i for i in range(66, 89)]
    + [i for i in range(90, 113)],
    "JacksonXml": range(1, 7),
    "Jsoup": range(1, 94),
    "JxPath": range(1, 23),
    "Lang": [1]
    + [i for i in range(3, 18)]
    + [i for i in range(19, 25)]
    + [i for i in range(26, 66) if i not in [2, 18, 25, 48]], # dropping non active bugs 
    "Math": range(1, 107),
    "Mockito": range(1, 39),
    "Time": [i for i in range(1, 21)] + [i for i in range(22, 28)],
}


for project in projects.keys():
    output_file = f"defects4j_bugs_{project}.jsonl"
    with open(output_file, "w") as outfile:
        for bug_id in projects[project]: 
            print(f"Processing {project} - Bug {bug_id}")
            bug_info = get_bug_info(project, bug_id)
            # Write the bug data to the JSONL file
            json.dump(bug_info, outfile)
            outfile.write("\n")
            outfile.flush()  # Ensure data is written to file promptly