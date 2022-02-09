@Library(['srePipeline']) _

// --------------------------------------------
// Refer to Pipeline docs for options used in mysettings
// https://wwwin-github.cisco.com/pages/eti/sre-pipeline-library
// --------------------------------------------

def pipelinesettings = [
  deploy: [
    [name: "fledge" ],                          // Containers to publish
    [name: "fledge-worker-allinone-cpu" ]       // Containers to publish
  ],

  tagversion: "${env.BUILD_ID}",     // Docker tag version
  chart: "deployment/helm-chart",    // Location of helm chart
  kubeverify: "fledge",              // Deploy verification name
  namespace: 'fledge',               // k8s namespace
  appname: 'fledge',                 // Deployment appname

  prepare: 1,                                   // GIT Clone
  unittest: 1,                                  // Unit-test
  build: 1,                                     // Build container
  buildMultipleContainerImages: 1,              // Build Multiple Container Images
  executeCC: 1,                                 // Generate Code Coverage report
  lint: 1,                                      // GO Lint
  // TODO: sonarQube fails at the moment. Enable it after a fix is found
  sonarQube: 0,                                 // SonarQube scan
  publishContainer: 1,                          // Publish container
  ecr: 1,                                       // Publish container to Private ECR
  ciscoContainer: 0,                            // Publish container to containers.cisco.com
  pushPublicRegistryOnTag: 1,                   // Publish container to Public ECR on tag
  publishHelm: 1,                               // Stage HELM CREATE
  deployHelm: 1,                                // Stage DEPLOY k8s
  // artifactory: 1,                               // Use Artifactory creds
  // stricterCCThreshold: 90.0,                    // Fail builds for Code Coverage below 90%
  awsLoginType:  "dynamic",
  secretsVaultAppname: "fledge",
]

srePipeline( pipelinesettings )
