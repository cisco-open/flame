@Library(['srePipeline']) _

// --------------------------------------------
// Refer to Pipeline docs for options used in mysettings
// https://wwwin-github.cisco.com/pages/eti/sre-pipeline-library
// --------------------------------------------

def pipelinesettings = [
  deploy: [
    [name: "fledge" ]                // Containers to publish
  ],

  tagversion: "${env.BUILD_ID}",                // Docker tag version
  chart: "deployment/helm-chart",               // Location of helm chart
  kubeverify: "fledge",              // Deploy verification name
  namespace: 'fledge',                      // k8s namespace
  appname: 'fledge',                 // Deployment appname

  prepare: 1,                                   // GIT Clone
  unittest: 1,                                  // Unit-test
  build: 1,                                     // Build container
  executeCC: 1,                                 // Generate Code Coverage report
  // TODO: can't enable lint and sonarqube yet because there are many errors
  // TODO: enable them after all errors are handled
  // lint: 1,                                      // GO Lint
  // sonarQube: 1,                                 // SonarQube scan
  publishContainer: 1,                          // Publish container
  ecr: 1,                                       // Publish container to Private ECR
  ciscoContainer: 1,                            // Publish container to containers.cisco.com
  // dockerHub: 1,                                 // Publish container to dockerhub.cisco.com
  pushPublicRegistryOnTag: 1,                   // Publish container to Public ECR on tag
  // forceCorona: 1,                            // Force Corona Scan on any branch
  // corona: [                                     // Corona paramters
  //   imageName: "sre-go-helloworld",             // Corona Image Name
  //   releaseID: "73243",                         // Corona Release ID
  //   productID: "6726",                          // Corona Project ID
  //   csdlID: "84720",                            // Corona CSDL ID
  //   securityContact: "sraradhy@cisco.com",      // Corona Security Contact
  //   engineeringContact: "sraradhy@cisco.com",   // Corona Engineering Contact
  //   imageAdmins: "sraradhy,jegarnie",           // Corona Image Admins
  // ],
  // forceBlackduck: 1,                         // Force Blackduck Scan on any branch
  // blackduck: [
  //   email: "eti-sre-admins@cisco.com",
  // ],                                            // Blackduck Open Source Scan
  publishHelm: 1,                               // Stage HELM CREATE
  deployHelm: 1,                                // Stage DEPLOY k8s
  // artifactory: 1,                               // Use Artifactory creds
  // stricterCCThreshold: 90.0,                    // Fail builds for Code Coverage below 90%
  awsLoginType:  "dynamic",
  secretsVaultAppname: "fledge",
]

srePipeline( pipelinesettings )

