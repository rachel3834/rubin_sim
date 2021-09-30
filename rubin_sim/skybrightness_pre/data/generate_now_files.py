from generate_sky import generate_sky


if __name__ == '__main__':
    # A quick run to generate files for Tiago so it will run present day.
    generate_sky(mjd0=59488, mjd_max=59560+10.)
