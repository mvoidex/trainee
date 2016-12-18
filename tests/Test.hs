module Main (
	main
	) where

import Control.Concurrent
import Control.Monad
import Control.Monad.State

import Numeric.Trainee.Neural
import Numeric.LinearAlgebra

test ∷ IO (Net Double)
test = net $ input 2 ⭃ fc sigma 5 ⭃ fc sigma 3 ⭃ fc sigma 1

fn ∷ Double → Double → Double
fn x y = sin x * 0.01 + cos (y / 3) * 0.02

examples ∷ [(Vector Double, Vector Double)]
examples = [(vector [x, y], vector [fn x y]) | x ← [-1.0, -0.95 .. 1.0], y ← [-1.0, -0.95 .. 1.0]]

main ∷ IO ()
main = do
	n ← test
	void $ runLearnT n $ forever $ do
		e ← trainUntil 0.1 100 50 0.01 squared examples
		liftIO $ print e
		-- liftIO $ threadDelay 100000
