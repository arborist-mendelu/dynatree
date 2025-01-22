eval "$(/home/marik/miniforge3/condabin/conda shell.bash hook)"
/home/marik/miniforge3/condabin/conda activate dynatree
#/home/marik/miniforge3/envs/dynatree/bin/solara run solara_app.py --no-open --host=um-bc201.mendelu.cz
SOLARA_APP=solara_app.py /home/marik/miniforge3/envs/dynatree/bin/gunicorn --workers 4 --threads=20 -b 0.0.0.0:5000 app:app

