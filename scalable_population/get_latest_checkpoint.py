import sys, os
if __name__ == '__main__':
    logdir = sys.argv[1]

    if True:
        os.system("mkdir agents/" + logdir)
        os.system("scp thunder.afrl.hpc.mil:/p/home/jasekp/scalable_population/agents/" + logdir + "/checkpoint" + " agents/" + logdir + "/.")

        with open('agents/' + logdir + '/checkpoint', 'r') as f:
            latest_model = f.readlines()[-1].split('"')[1]
            print(latest_model)
            os.system("scp thunder.afrl.hpc.mil:/p/home/jasekp/scalable_population/agents/" + logdir + "/" + latest_model + "* agents/" + logdir + "/.")

        os.system("scp thunder.afrl.hpc.mil:/p/home/jasekp/scalable_population/agents/" + logdir + "/events*" + " agents/" + logdir + "/.")
    else:
        os.system("mkdir old_agents/" + logdir)
        os.system("scp thunder.afrl.hpc.mil:/p/home/jasekp/scalable_population/agents/" + logdir + "/checkpoint" + " old_agents/" + logdir + "/.")

        with open('old_agents/' + logdir + '/checkpoint', 'r') as f:
            latest_model = f.readlines()[-1].split('"')[1]
            print(latest_model)
            os.system("scp thunder.afrl.hpc.mil:/p/home/jasekp/scalable_population/agents/" + logdir + "/" + latest_model + "* old_agents/" + logdir + "/.")

        os.system("scp thunder.afrl.hpc.mil:/p/home/jasekp/scalable_population/agents/" + logdir + "/events*" + " old_agents/" + logdir + "/.")

