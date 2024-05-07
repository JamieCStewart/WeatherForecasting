#!/usr/bin/env python

# Import required python modules
import ftplib
import os
import calendar
import configparser

# Manually defne the resolution
res = 25
start_year = 2021
end_year = 2022

# Read the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Get the username and password from the configuration file
username = config['credentials']['username']
password = config['credentials']['password']

# Define the local directory name to put data in
ddir = f"./{res}km/Rainfall"  

# If directory doesn't exist make it
if not os.path.isdir(ddir):
    os.makedirs(ddir)  

# Change the local directory to where you want to put the data
os.chdir(ddir)

# login to FTP
ftp_server = "ftp.ceda.ac.uk"

f = ftplib.FTP(ftp_server)
f.login(username, password)

# Set the remote directory
remote_dir = (
            f'/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid'
            f'/v1.2.0.ceda/{res}km/rainfall/day/v20230328'
            )

# Change the remote directory
f.cwd(remote_dir)

# Loop through years
for year in range(start_year, end_year+1):
    print(year)
    # Loop through months
    for month in range(1, 13):

        # Define filename
        start_date = f"{year}{month:02d}01"
        last_day = calendar.monthrange(year, month)[1]
        end_date = f"{year}{month:02d}{last_day}"
        file = (f"rainfall_hadukgrid_uk_{res}km_day_"
                f"{start_date}-{end_date}.nc")

        # Get the remote file to the local directory
        f.retrbinary("RETR %s" % file, open(file, "wb").write)

# Close FTP connection
f.quit()
