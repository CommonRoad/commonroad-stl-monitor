## Install all needed packages
- Activate your conda env.
- Go to  ../stl_crmonitor and run 
``
pip install -r requirements.txt
``

## Download rtamt library 
- Follow the intruction in this link https://github.com/nickovic/rtamt to install all the needed packages, clone the repository, install and test its installaion.
- Remark: If you don't have sudo rights for installaion, you can simply run `pip3 install .`

## Run the tests
- Run 
``
pytest crmonitor/tests
``
