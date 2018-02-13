#########
# Intro #
#########
echo "This script will install CDMS into a virtual environment.
Make sure that you are not already sourced in another environment."
echo ""

echo -n "Do you wish to proceed? (y/n): "
read proceed

if [ "$proceed" = "n" ]; then
    echo "exiting..."
    exit
else
    echo -n "Pick name for install directory: "
    read dir
    echo "starting install..."
    basedir=$(pwd)
fi

##################################
# Create the virtual environment #
##################################
cd $HOME

# Make sure virtualenv exists
if hash virtualenv 2> /dev/null; then
    virtualenv $dir
else
    echo "Need to install virtualenv"
    echo -n "Install it now? (y/n): "
    read install

    if [ "$install" = "n" ]; then
        echo "exiting..."
        exit
    else
        echo "installing virtualenv..."
        pip install virtualenv
        virtualenv $dir
    fi
fi

# source the VE
source $HOME/$dir/bin/activate

######################
# Install everything #
######################
pip install --upgrade pip
pip install numpy
pip install matplotlib
pip install scipy
pip install tensorflow-gpu
pip install keras
