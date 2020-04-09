# Import the Portal object.
import geni.portal as portal
# Import the ProtoGENI library.
import geni.rspec.pg as pg
import geni.rspec.igext as IG

# Create a portal context.
pc = portal.Context()

pc.defineParameter( "corecount", "Number of cores in each node (1 or more).  NB: Make certain your requested cluster can supply this quantity.", portal.ParameterType.INTEGER, 4 )
pc.defineParameter( "ramsize", "MB of RAM in each node (2048 or more).  NB: Make certain your requested cluster can supply this quantity.", portal.ParameterType.INTEGER, 56000 )
pc.defineParameter( "hdsize", "GB of hard drive (2 or more).  NB: Make certain your requested cluster can supply this quantity.", portal.ParameterType.INTEGER, 32 )
params = pc.bindParameters()

# Create a Request object to start building the RSpec.
request = pc.makeRequestRSpec()

tourDescription = \
"""
Glaucoma neural network on Ubuntu 18.04.
"""

#
# Setup the Tour info with the above description and instructions.
#  
tour = IG.Tour()
tour.Description(IG.Tour.TEXT,tourDescription)
request.addTour(tour)

prefixForIP = "192.168.1."

#link = request.LAN("lan")

node = request.XenVM("glaunet1")

node.cores = params.corecount
node.ram = params.ramsize
node.disk_image = "urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU18-64-STD"
node.Attribute('XEN_EXTRAFS',str(hdsize))

#iface = node.addInterface("if1")
#iface.component_id = "eth1"
#iface.addAddress(pg.IPv4Address(prefixForIP + "1", "255.255.255.0"))
node.routable_control_ip = "true" 
#link.addInterface(iface)  

# Set scripts in the repository executable and readable.
node.addService(pg.Execute(shell="sh", command="sudo find /local/repository/ -type f -iname \"*.sh\" -exec chmod 755 {} \;"))

# Run setup script(s)
node.addService(pg.Execute(shell="sh", command="sudo /local/repository/singularity/singularitySetup.sh"))
  
# Print the RSpec to the enclosing page.
pc.printRequestRSpec(request)
