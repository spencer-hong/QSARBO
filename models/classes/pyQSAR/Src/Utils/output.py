'''
Output

DESCRIPTION
    Provides output functionality.
'''

# Imports
import time
import platform

# Functions
def print_startBanner():
    '''
    Print start calculation banner for pyQSAR.
    '''

    # Variables

    # Print start
    print("========================================")
    print("               ___  ____    _    ____   ")
    print("  _ __  _   _ / _ \/ ___|  / \  |  _ \  ")
    print(" | '_ \| | | | | | \___ \ / _ \ | |_) | ")
    print(" | |_) | |_| | |_| |___) / ___ \|  _ <  ")
    print(" | .__/ \__, |\__\_\____/_/   \_\_| \_\ ")
    print(" |_|    |___/                           ")
    print("========================================")

    # Print system information
    print("OS: " + platform.platform())

    # Print time stamp
    print("Start Time: " + time.asctime())
    print("")

def print_endBanner():
    '''
    Print the end banner for a calculation.
    '''

    # Start block
    print("========================================")

    # Print time
    print("End Time: " + time.asctime())

# Main
if (__name__ == '__main__'):
    pass
