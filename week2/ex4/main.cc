#include <cstdio>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "t1.h"

#include <TMath.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TCanvas.h> 
#include <TLorentzVector.h>



//------------------------------------------------------------------------------
// Particle Class
//
class Particle{

	public:
	Particle();
	// RESOLVED : Create an additional constructor that takes 4 arguments --> the 4-momentum
	Particle(double p[4]);
	double   pt, eta, phi, E, m, p[4];
	void     p4(double, double, double, double);
	void     print();
	void     setMass(double);
	double   sintheta();
};

//------------------------------------------------------------------------------

//*****************************************************************************
//                                                                             *
//    MEMBERS functions of the Particle Class                                  *
//                                                                             *
//*****************************************************************************

//
//*** Default constructor ------------------------------------------------------
//
Particle::Particle()
{
	pt = eta = phi = E = m = 0.0;
	p[0] = p[1] = p[2] = p[3] = 0.0;
}

//*** Additional constructor ------------------------------------------------------
Particle::Particle(double p[4])
{ 
	for(int idx = 0; idx < 4; idx++)
		this->p[idx] = p[idx];
}

//
//*** Members  ------------------------------------------------------
//
double Particle::sintheta()
{
	return sin(2.0 * atan(exp(-1.0 * this->p[2])));
}

void Particle::p4(double pT, double eta, double phi, double energy)
{
	this->p[0] = energy;
	this->p[1] = pT * cos(phi);
	this->p[2] = pT * sin(phi);
	this->p[3] = pT * sinh(eta);
}

void Particle::setMass(double mass)
{
	this->m = mass;
}

//
//*** Prints 4-vector ----------------------------------------------------------
//
void Particle::print()
{
	std::cout << "\t(" << p[0] <<",\t" << p[1] <<",\t"<< p[2] <<",\t"<< p[3] << ")" << "  " <<  sintheta() << std::endl;
}

//------------------------------------------------------------------------------
// Lepton Class
//
class Lepton: public Particle
{
	public:
	float charge; 
	Lepton(): Particle() {}
	Lepton(double pT, double eta, double phi, double energy, float charge): Particle() 
	{
		p4(pT, eta, phi, energy); 
		this->charge = charge;
	}
	void set_charge(float charge);
	float get_charge();
};

void Lepton::set_charge(float charge)
{
	this->charge = charge;
}

float Lepton::get_charge()
{
	return this->charge;
}

//------------------------------------------------------------------------------
// Jets Class
//
class Jets: public Particle
{
	public: 
		float flavor;
		Jets(): Particle() {}
		Jets(double pT, double eta, double phi, double energy, float flavor): Particle() 
		{
			p4(pT, eta, phi, energy);  
			this->flavor = flavor;
		}
		void set_flavor(float flavor);
		float get_flavor();

};

void Jets::set_flavor(float flavor)
{
	this->flavor = flavor;
}

float Jets::get_flavor()
{
	return this->flavor;
}

int main() {
	
	/* ************* */
	/* Input Tree   */
	/* ************* */

	TFile *f      = new TFile("input.root","READ");
	TTree *t1 = (TTree*)(f->Get("t1"));

	// Total number of events in ROOT tree
	Long64_t nentries = t1->GetEntries();

	for (Long64_t jentry=0; jentry<100;jentry++)
 	{
		t1->GetEntry(jentry);
		// Read the variables from the ROOT tree branches
		t1->SetBranchAddress("lepPt",&lepPt);
		t1->SetBranchAddress("lepEta",&lepEta);
		t1->SetBranchAddress("lepPhi",&lepPhi);
		t1->SetBranchAddress("lepE",&lepE);
		t1->SetBranchAddress("lepQ",&lepQ);
		
		t1->SetBranchAddress("njets",&njets);
		t1->SetBranchAddress("jetPt",&jetPt);
		t1->SetBranchAddress("jetEta",&jetEta);
		t1->SetBranchAddress("jetPhi",&jetPhi);
		t1->SetBranchAddress("jetE", &jetE);
		t1->SetBranchAddress("jetHadronFlavour",&jetHadronFlavour);
		std::cout<<" Event "<< jentry <<std::endl;	

		// Lepton loop
		for (int i = 0; i < std::size(lepPt); i++)
		{
			if (lepE[i] == 0)
				break;

			Lepton lepton = Lepton(
				lepPt[i],
				lepEta[i],
				lepPhi[i],
				lepE[i],
				lepQ[i]
			);
			std::cout << "\tLepton " << i + 1 << std::endl;
			lepton.print();
			std::cout << "\tCharge: " << lepton.get_charge() << std::endl << std::endl;

		}

		// Jet loop
		for (int i = 0; i < std::size(jetPt); i++)
		{
			if (jetE[i] == 0)
				break;

			Jets jet = Jets(
				jetPt[i],
				jetEta[i],
				jetPhi[i],
				jetE[i],
				jetHadronFlavour[i]
			);
			std::cout << "\tJet " << i + 1 << std::endl;
			jet.print();
			std::cout << "\tHadron Flavor: " << jet.get_flavor() << std::endl << std::endl;

		}
	} // Loop over all events

  	return 0;
}