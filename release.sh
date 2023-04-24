echo
echo "----------------- updating poetry... -----------------------"
poetry update
poetry install --no-root --extras full

echo "----------------- updated -----------------------"
echo; echo

echo
echo "----------------- testing... -----------------------"
poetry run pytest src tests --cov=src --doctest-modules  --cov-report term-missing
echo "----------------- tested -----------------------"
echo; echo

#echo
#echo "----------------- gh workflow testing... -----------------------"
#read -p "press enter"
# sudo systemctl enable docker
# sudo systemctl start docker
# coverage xml
# act -j build
# sudo systemctl stop docker
# sudo systemctl disable docker
#echo "----------------- gh workflow -----------------------"
#echo; echo


echo
echo "----------------- docs/black... -----------------------"
read -p "press enter"
#################################################################################
#################################################################################
echo ">>>>>>   install project package for IDE class hierarchy <<<<<<<<" 
echo "          (to remove duplicates from IDE class hierarchy)"
source /home/davi/.cache/pypoetry/virtualenvs/hdict-ZUd5y1Rg-py3.10/bin/activate
pip install .
#################################################################################
#################################################################################
rm docs -rf
poetry run black -l200 src/ tests/
poetry run pdoc --html --force sortedness -o docs
mv docs/sortedness/* docs/
rm docs/sortedness -rf
git add docs
echo "----------------- docs/black done -----------------------"
echo; echo

echo "---------------- readme ----------------"
poetry run autoreadme -i README-edit.md -s examples/ -o README.md
echo "---------------- readme done ----------------"
echo; echo

#################################################################################
#################################################################################
echo ">>>>>>   uninstall project package for IDE class hierarchy <<<<<<<<" 
pip uninstall hdict -y
deactivate
#################################################################################
#################################################################################
echo; echo



echo "×××××××××××××××× version bump ××××××××××××××××"
read -p "press enter"
poetry version patch
echo "--------------- version bumped --------------"
echo; echo

echo "------------------ current status -----------------------"
git status
echo "------------------ current status shown-----------------"
echo; echo

echo "------------------ commit --------------------"
read -p "press enter"
git commit -am "Release"
echo "------------------ commited --------------------"
echo; echo

echo "------------------ new status... -----------------------"
read -p "press enter"
git status
echo "------------------ new status shown --------------------"
echo; echo

echo "------------------- tag ----------------------"
read -p "press enter"
git tag "v$(poetry version | cut -d' ' -f2)" -m "Release v$(poetry version | cut -d' ' -f2)"
echo "------------------- tagged ----------------------"
echo; echo

echo "------------------- push ----------------------"
read -p "press enter"
git push origin main "v$(poetry version | cut -d' ' -f2)"
echo "------------------- pushed ----------------------"
echo; echo

echo "------------------- publish ----------------------"
poetry publish --build
