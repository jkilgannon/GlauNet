# Import the Portal object.
import geni.portal as portal
# Import the ProtoGENI library.
import geni.rspec.pg as pg
import geni.rspec.igext as IG

# Create a portal context.
pc = portal.Context()

params = pc.bindParameters()

# Create a Request object to start building the RSpec.
request = pc.makeRequestRSpec()

tourDescription = \
"""
This profile provides a Slurm and Open MPI cluster installed on Ubuntu 18.04.
"""

#
# Setup the Tour info with the above description and instructions.
#  
tour = IG.Tour()
tour.Description(IG.Tour.TEXT,tourDescription)
request.addTour(tour)

prefixForIP = "192.168.1."

link = request.LAN("lan")

node = request.XenVM("glaunet1")

node.cores = 4
node.ram = 56000
node.disk_image = "urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU18-64-STD"

iface = node.addInterface("if1")
iface.component_id = "eth1"
iface.addAddress(pg.IPv4Address(prefixForIP + "1", "255.255.255.0"))
node.routable_control_ip = "true" 
link.addInterface(iface)  

# Set scripts in the repository executable and readable.
node.addService(pg.Execute(shell="sh", command="sudo find /local/repository/ -type f -iname \"*.sh\" -exec chmod 755 {} \;"))
  
# Print the RSpec to the enclosing page.
pc.printRequestRSpec(request)
