if [ ! -z "$1" ]
	then
	export KERASTUNER_TUNER_ID="tuner$1"
fi
. ./venv/Scripts/activate
python main.py